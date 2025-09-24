import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import rasterio
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

class AdvancedSAMRemoteSensing:
    """结合SAM和CLIP的先进遥感解译系统"""
    
    def __init__(self):
        # SAM模型
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
        self.sam.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        # CLIP模型用于零样本分类
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = AutoModelForZeroShotImageClassification.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
        # 遥感类别描述
        self.rs_classes = [
            "建筑物屋顶",
            "道路或停车场", 
            "绿色植被或树木",
            "水体或河流",
            "裸露土地",
            "农田",
            "太阳能板",
            "运动场"
        ]
        
    def segment_and_classify(self, image_path):
        """分割并分类遥感图像"""
        # 读取图像
        with rasterio.open(image_path) as src:
            image = src.read()[:3]  # RGB波段
            image = np.transpose(image, (1, 2, 0))
            
        # 使用SAM生成掩码
        masks = self.mask_generator.generate(image)
        
        # 对每个掩码进行分类
        classified_masks = []
        for mask_data in masks:
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']
            
            # 提取掩码区域
            x, y, w, h = bbox
            cropped = image[y:y+h, x:x+w]
            
            # 使用CLIP进行零样本分类
            inputs = self.clip_processor(
                text=self.rs_classes,
                images=cropped,
                return_tensors="pt",
                padding=True
            )
            
            outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            predicted_class = self.rs_classes[probs.argmax()]
            
            classified_masks.append({
                'mask': mask,
                'class': predicted_class,
                'confidence': probs.max().item(),
                'area': mask_data['area']
            })
            
        return classified_masks