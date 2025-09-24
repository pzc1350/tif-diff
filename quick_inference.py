import torch
import numpy as np
import rasterio
from pathlib import Path
import segmentation_models_pytorch as smp
from PIL import Image
import matplotlib.pyplot as plt

class QuickRemoteSensingPredictor:
    """快速遥感图像预测器，使用预训练模型"""
    
    def __init__(self, model_type='unet', num_classes=6):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        
        # 使用预训练的编码器
        self.model = self._get_pretrained_model(model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # 类别颜色映射
        self.colors = {
            0: [0, 0, 0],       # 背景 - 黑色
            1: [255, 0, 0],     # 建筑物 - 红色
            2: [128, 128, 128], # 道路 - 灰色
            3: [0, 255, 0],     # 植被 - 绿色
            4: [0, 0, 255],     # 水体 - 蓝色
            5: [165, 42, 42]    # 裸土 - 棕色
        }
        
    def _get_pretrained_model(self, model_type):
        """获取预训练模型"""
        if model_type == 'unet':
            # 使用ImageNet预训练的ResNet34作为编码器
            model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes
            )
        elif model_type == 'deeplabv3':
            model = smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes
            )
        elif model_type == 'fpn':
            model = smp.FPN(
                encoder_name="efficientnet-b0",
                encoder_weights="imagenet",
                in_channels=3,
                classes=self.num_classes
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
        return model
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 归一化到0-1
        if image.max() > 1:
            image = image / 255.0
            
        # 标准化（ImageNet的均值和标准差）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        return image
    
    def predict_tif(self, tif_path, patch_size=512, overlap=64):
        """对GeoTIFF图像进行预测（支持大图像）"""
        with rasterio.open(tif_path) as src:
            # 读取图像和地理信息
            image = src.read()
            profile = src.profile
            transform = src.transform
            
        # 转换为HWC格式
        image = np.transpose(image[:3], (1, 2, 0))  # 只取前3个波段
        h, w = image.shape[:2]
        
        # 如果图像太大，使用滑动窗口
        if h > patch_size or w > patch_size:
            prediction = self._predict_large_image(image, patch_size, overlap)
        else:
            prediction = self._predict_patch(image)
            
        return prediction, profile
    
    def _predict_patch(self, patch):
        """预测单个图像块"""
        # 预处理
        patch = self.preprocess_image(patch)
        
        # 转换为tensor
        patch_tensor = torch.from_numpy(patch).float()
        patch_tensor = patch_tensor.permute(2, 0, 1).unsqueeze(0)
        patch_tensor = patch_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(patch_tensor)
            prediction = output.argmax(dim=1).squeeze().cpu().numpy()
            
        return prediction
    
    def _predict_large_image(self, image, patch_size, overlap):
        """使用滑动窗口预测大图像"""
        h, w = image.shape[:2]
        stride = patch_size - overlap
        
        # 创建输出数组
        prediction = np.zeros((h, w), dtype=np.uint8)
        count = np.zeros((h, w), dtype=np.float32)
        
        # 滑动窗口
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # 提取patch
                patch = image[y:y+patch_size, x:x+patch_size]
                
                # 预测
                patch_pred = self._predict_patch(patch)
                
                # 累加预测结果
                prediction[y:y+patch_size, x:x+patch_size] += patch_pred
                count[y:y+patch_size, x:x+patch_size] += 1
                
        # 处理边缘
        # ... (简化处理，实际使用时需要补充)
        
        # 平均预测结果
        prediction = (prediction / (count + 1e-6)).astype(np.uint8)
        
        return prediction
    
    def visualize_prediction(self, image, prediction, save_path=None):
        """可视化预测结果"""
        # 创建彩色标签图
        h, w = prediction.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label, color in self.colors.items():
            color_mask[prediction == label] = color
            
        # 绘图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        if image.shape[2] > 3:
            image = image[:, :, :3]
        axes[0].imshow(image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 预测结果
        axes[1].imshow(color_mask)
        axes[1].set_title('预测结果')
        axes[1].axis('off')
        
        # 叠加显示
        overlay = image.copy()
        overlay = (overlay * 0.6 + color_mask / 255.0 * 0.4)
        axes[2].imshow(overlay)
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        return color_mask

# 使用示例
def quick_predict(input_path, output_path=None, model_type='unet', visualize=True):
    """快速预测函数"""
    print(f"正在加载 {model_type} 模型...")
    predictor = QuickRemoteSensingPredictor(model_type=model_type)
    
    print(f"正在处理图像: {input_path}")
    prediction, profile = predictor.predict_tif(input_path)
    
    # 保存预测结果
    if output_path:
        profile.update({
            'count': 1,
            'dtype': rasterio.uint8
        })
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)
        print(f"预测结果已保存到: {output_path}")
    
    # 可视化
    if visualize:
        with rasterio.open(input_path) as src:
            image = src.read()
            image = np.transpose(image[:3], (1, 2, 0))
            
        predictor.visualize_prediction(image, prediction, 
                                     save_path='prediction_result.png')
    
    return prediction

if __name__ == "__main__":
    # 示例用法
    input_tif = "data/sample_image.tif"
    output_tif = "data/prediction.tif"
    
    # 使用预训练的UNet进行快速预测
    prediction = quick_predict(input_tif, output_tif, model_type='unet')