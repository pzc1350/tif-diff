import torch
import rasterio
import numpy as np
import argparse
from pathlib import Path
import yaml

from src.models.unet import UNet
from src.utils.visualization import visualize_prediction

def predict_single_image(model, image_path, device, transform=None):
    """对单张遥感图像进行预测"""
    
    # 读取遥感图像
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile
        
    # 转换为(H, W, C)格式
    image = np.transpose(image, (1, 2, 0))
    
    # 数据预处理
    if transform:
        augmented = transform(image=image)
        image = augmented['image']
    
    # 添加批次维度并移至设备
    image = image.unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1).squeeze().cpu().numpy()
    
    return prediction, profile

def save_prediction(prediction, output_path, profile):
    """保存预测结果为GeoTIFF"""
    
    # 更新配置
    profile.update({
        'count': 1,
        'dtype': rasterio.uint8
    })
    
    # 写入文件
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(prediction.astype(np.uint8), 1)

def main():
    parser = argparse.ArgumentParser(description='遥感图像推理')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--input_path', type=str, required=True,
                        help='输入图像路径')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出路径')
    parser.add_argument('--visualize', action='store_true',
                        help='是否可视化结果')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和配置
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['config']
    
    # 创建模型
    model = UNet(
        n_channels=config['model']['in_channels'],
        n_classes=config['data']['num_classes']
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 进行预测
    prediction, profile = predict_single_image(
        model, args.input_path, device
    )
    
    # 保存结果
    if args.output_path is None:
        input_path = Path(args.input_path)
        output_path = input_path.parent / f"{input_path.stem}_prediction.tif"
    else:
        output_path = args.output_path
        
    save_prediction(prediction, output_path, profile)
    print(f"Prediction saved to: {output_path}")
    
    # 可视化
    if args.visualize:
        visualize_prediction(args.input_path, prediction, config['classes'])

if __name__ == "__main__":
    main()