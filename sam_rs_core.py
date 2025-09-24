"""
遥感图像SAM处理核心模块

本模块实现了基于Segment Anything Model (SAM) 的遥感图像智能解译功能，
包括图像分割、语义分类、光谱指数计算等核心功能。

主要类：
    - RemoteSensingConfig: 配置类，管理所有处理参数
    - SpectralIndicesCalculator: 光谱指数计算器
    - RemoteSensingSAM: 核心处理类，集成SAM和CLIP进行遥感图像解译

作者: pzc1350
日期: 2025-09-24
版本: 0.1.0
"""

import torch
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass, field
from tqdm import tqdm

# SAM相关导入
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import CLIPProcessor, CLIPModel
import supervision as sv


@dataclass
class RemoteSensingConfig:
    """
    遥感处理配置类
    
    该类定义了遥感图像处理的所有配置参数，包括模型设置、处理参数、
    类别定义等。使用dataclass简化配置管理。
    
    Attributes:
        sam_checkpoint (str): SAM模型权重文件路径
        sam_model_type (str): SAM模型类型，可选 "vit_h", "vit_l", "vit_b"
        device (str): 计算设备，"cuda" 或 "cpu"
        spectral_indices (List[str]): 需要计算的光谱指数列表
        tile_size (int): 分块处理时的块大小（像素）
        overlap (int): 分块之间的重叠区域大小（像素）
        min_mask_region_area (int): 最小掩码区域面积（像素）
        rs_classes (Dict[str, List[str]]): 遥感类别定义字典
    """
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"  # 可选: vit_h, vit_l, vit_b
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 遥感特定参数
    spectral_indices: List[str] = field(default_factory=list)
    tile_size: int = 1024  # 分块大小
    overlap: int = 128  # 重叠区域
    min_mask_region_area: int = 100  # 最小掩码区域
    
    # 类别定义
    rs_classes: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if not self.spectral_indices:
            # 默认计算的光谱指数
            self.spectral_indices = ['NDVI', 'NDWI', 'NDBI', 'SAVI']
            
        if not self.rs_classes:
            # 默认的遥感类别定义
            self.rs_classes = {
                'urban': ['building', 'road', 'parking lot', 'concrete structure'],
                'vegetation': ['forest', 'grassland', 'cropland', 'park'],
                'water': ['river', 'lake', 'pond', 'ocean'],
                'bare_land': ['soil', 'sand', 'rock', 'desert']
            }


class SpectralIndicesCalculator:
    """
    光谱指数计算器
    
    提供各种常用遥感光谱指数的计算方法。所有方法都是静态方法，
    可以直接调用而无需实例化类。
    """
    
    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        计算归一化植被指数 (NDVI)
        
        NDVI = (NIR - Red) / (NIR + Red)
        值域：[-1, 1]，越接近1表示植被越茂密
        
        Args:
            nir: 近红外波段数据
            red: 红光波段数据
            
        Returns:
            NDVI数组
        """
        # 添加小值避免除零
        return (nir - red) / (nir + red + 1e-8)
    
    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        计算归一化水体指数 (NDWI)
        
        NDWI = (Green - NIR) / (Green + NIR)
        值域：[-1, 1]，正值通常表示水体
        
        Args:
            green: 绿光波段数据
            nir: 近红外波段数据
            
        Returns:
            NDWI数组
        """
        return (green - nir) / (green + nir + 1e-8)
    
    @staticmethod
    def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        计算归一化建筑指数 (NDBI)
        
        NDBI = (SWIR - NIR) / (SWIR + NIR)
        用于识别建筑区域
        
        Args:
            swir: 短波红外波段数据
            nir: 近红外波段数据
            
        Returns:
            NDBI数组
        """
        return (swir - nir) / (swir + nir + 1e-8)
    
    @staticmethod
    def calculate_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        计算土壤调节植被指数 (SAVI)
        
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        适用于植被稀疏地区，减少土壤背景影响
        
        Args:
            nir: 近红外波段数据
            red: 红光波段数据
            L: 土壤调节因子，默认0.5
            
        Returns:
            SAVI数组
        """
        return ((nir - red) / (nir + red + L)) * (1 + L)
    
    @staticmethod
    def calculate_all_indices(image_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算所有可用的光谱指数
        
        根据输入的波段数据，自动计算所有可能的光谱指数
        
        Args:
            image_data: 包含各波段数据的字典，键为波段名称
            
        Returns:
            包含所有计算出的光谱指数的字典
        """
        indices = {}
        
        # 计算NDVI
        if 'nir' in image_data and 'red' in image_data:
            indices['ndvi'] = SpectralIndicesCalculator.calculate_ndvi(
                image_data['nir'], image_data['red']
            )
            
        # 计算NDWI
        if 'green' in image_data and 'nir' in image_data:
            indices['ndwi'] = SpectralIndicesCalculator.calculate_ndwi(
                image_data['green'], image_data['nir']
            )
            
        # 计算NDBI
        if 'swir' in image_data and 'nir' in image_data:
            indices['ndbi'] = SpectralIndicesCalculator.calculate_ndbi(
                image_data['swir'], image_data['nir']
            )
            
        # 计算SAVI
        if 'nir' in image_data and 'red' in image_data:
            indices['savi'] = SpectralIndicesCalculator.calculate_savi(
                image_data['nir'], image_data['red']
            )
            
        return indices


class RemoteSensingSAM:
    """
    遥感SAM处理主类
    
    该类集成了SAM（Segment Anything Model）和CLIP模型，专门用于处理
    遥感图像。支持多光谱数据、大图像分块处理、零样本分类等功能。
    
    主要功能：
        1. 遥感图像自动分割
        2. 基于CLIP的零样本语义分类
        3. 光谱指数辅助分类
        4. 大图像分块处理
        5. 多格式数据支持
    """
    
    def __init__(self, config: RemoteSensingConfig):
        """
        初始化RemoteSensingSAM
        
        Args:
            config: 遥感处理配置对象
        """
        self.config = config
        
        # 初始化SAM模型
        print(f"加载SAM模型: {config.sam_model_type}")
        self.sam = sam_model_registry[config.sam_model_type](
            checkpoint=config.sam_checkpoint
        )
        self.sam.to(device=config.device)
        
        # 配置自动掩码生成器
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,  # 每边采样点数
            pred_iou_thresh=0.86,  # 预测IoU阈值
            stability_score_thresh=0.92,  # 稳定性分数阈值
            crop_n_layers=1,  # 裁剪层数
            crop_n_points_downscale_factor=2,  # 裁剪点下采样因子
            min_mask_region_area=config.min_mask_region_area  # 最小掩码区域
        )
        
        # 初始化CLIP模型用于语义理解
        print("加载CLIP模型...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(config.device)
        
    def process_remote_sensing_image(self, image_path: str) -> Dict:
        """
        处理遥感图像的主函数
        
        完整的处理流程包括：
        1. 读取多光谱数据
        2. 计算光谱指数
        3. 创建增强RGB图像
        4. SAM分割
        5. CLIP分类
        6. 生成语义分割图
        
        Args:
            image_path: 遥感图像文件路径
            
        Returns:
            包含处理结果的字典：
            {
                'segmentation_map': 语义分割图,
                'masks': 分类后的掩码列表,
                'spectral_indices': 光谱指数字典,
                'metadata': 图像元数据
            }
        """
        # 读取多光谱数据
        print(f"读取遥感图像: {image_path}")
        image_data = self._read_multispectral_image(image_path)
        
        # 计算光谱指数
        print("计算光谱指数...")
        spectral_indices = SpectralIndicesCalculator.calculate_all_indices(image_data)
        
        # 创建增强的RGB图像用于SAM处理
        enhanced_rgb = self._create_enhanced_rgb(image_data, spectral_indices)
        
        # 根据图像大小选择处理策略
        if enhanced_rgb.shape[0] > self.config.tile_size or enhanced_rgb.shape[1] > self.config.tile_size:
            # 大图像分块处理
            masks = self._process_large_image(enhanced_rgb)
        else:
            # 小图像直接处理
            masks = self.mask_generator.generate(enhanced_rgb)
        
        # 对每个掩码进行语义分类
        print("进行语义分类...")
        classified_masks = self._classify_masks(enhanced_rgb, masks, spectral_indices)
        
        # 生成最终的语义分割图
        segmentation_map = self._generate_segmentation_map(
            enhanced_rgb.shape[:2], classified_masks
        )
        
        return {
            'segmentation_map': segmentation_map,
            'masks': classified_masks,
            'spectral_indices': spectral_indices,
            'metadata': self._extract_metadata(image_path)
        }
    
    def _read_multispectral_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """
        读取多光谱遥感图像
        
        自动识别并读取各个波段，支持常见的多光谱传感器波段命名
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含各波段数据的字典
        """
        with rasterio.open(image_path) as src:
            # 读取所有波段
            bands = {}
            
            # 常见的多光谱波段映射
            # 适配Landsat、Sentinel等主流传感器
            band_names = {
                1: 'blue',      # 蓝光
                2: 'green',     # 绿光
                3: 'red',       # 红光
                4: 'nir',       # 近红外
                5: 'swir1',     # 短波红外1
                6: 'swir2'      # 短波红外2
            }
            
            # 读取每个波段
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                if i in band_names:
                    bands[band_names[i]] = band_data
                else:
                    bands[f'band_{i}'] = band_data
                    
            # 保存地理参考信息，后续导出时使用
            self.transform = src.transform
            self.crs = src.crs
            
        return bands
    
    def _create_enhanced_rgb(
        self, 
        image_data: Dict[str, np.ndarray], 
        spectral_indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        创建增强的RGB图像用于SAM处理
        
        将多光谱数据转换为RGB格式，并进行增强以提高分割效果
        
        Args:
            image_data: 包含各波段数据的字典
            spectral_indices: 计算出的光谱指数
            
        Returns:
            增强后的RGB图像 (H, W, 3)
        """
        # 获取RGB波段
        r = image_data.get('red', image_data.get('band_3', None))
        g = image_data.get('green', image_data.get('band_2', None))
        b = image_data.get('blue', image_data.get('band_1', None))
        
        if r is None or g is None or b is None:
            raise ValueError("无法找到RGB波段")
            
        # 堆叠为RGB图像
        rgb = np.stack([r, g, b], axis=-1)
        
        # 增强对比度以提高分割效果
        rgb = self._enhance_contrast(rgb)
        
        # 转换为uint8格式
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        增强图像对比度
        
        使用百分位数拉伸方法增强图像对比度，提高视觉效果
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像，值域[0, 1]
        """
        # 计算2%和98%分位数
        p2, p98 = np.percentile(image, (2, 98))
        # 拉伸到[0, 1]范围
        image = np.clip((image - p2) / (p98 - p2), 0, 1)
        return image
    
    def _process_large_image(self, image: np.ndarray) -> List[Dict]:
        """
        处理大尺寸图像（分块处理）
        
        将大图像分成重叠的块进行处理，然后合并结果。
        这样可以处理任意大小的遥感图像而不会内存溢出。
        
        Args:
            image: 输入的大尺寸图像
            
        Returns:
            所有检测到的掩码列表
        """
        h, w = image.shape[:2]
        tile_size = self.config.tile_size
        overlap = self.config.overlap
        stride = tile_size - overlap
        
        all_masks = []
        
        # 计算需要的分块数量
        n_tiles_h = (h - overlap) // stride + 1
        n_tiles_w = (w - overlap) // stride + 1
        
        print(f"图像大小: {h}x{w}, 分块数: {n_tiles_h}x{n_tiles_w}")
        
        # 使用进度条显示处理进度
        with tqdm(total=n_tiles_h * n_tiles_w, desc="处理图像块") as pbar:
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    # 计算当前块的坐标
                    y_start = i * stride
                    x_start = j * stride
                    y_end = min(y_start + tile_size, h)
                    x_end = min(x_start + tile_size, w)
                    
                    # 提取当前块
                    tile = image[y_start:y_end, x_start:x_end]
                    
                    # 在当前块上生成掩码
                    tile_masks = self.mask_generator.generate(tile)
                    
                    # 将掩码坐标转换回原图坐标系
                    for mask in tile_masks:
                        # 调整边界框坐标
                        mask['bbox'][0] += x_start
                        mask['bbox'][1] += y_start
                        mask['bbox'][2] += x_start
                        mask['bbox'][3] += y_start
                        
                        # 调整分割掩码
                        full_mask = np.zeros((h, w), dtype=bool)
                        full_mask[y_start:y_end, x_start:x_end] = mask['segmentation']
                        mask['segmentation'] = full_mask
                        
                    all_masks.extend(tile_masks)
                    pbar.update(1)
                    
        # 去除重复的掩码
        all_masks = self._remove_duplicate_masks(all_masks)
        
        return all_masks
    
    def _remove_duplicate_masks(self, masks: List[Dict]) -> List[Dict]:
        """
        去除重复的掩码
        
        基于IoU（交并比）去除重叠度过高的掩码，保留质量更好的掩码
        
        Args:
            masks: 原始掩码列表
            
        Returns:
            去重后的掩码列表
        """
        filtered_masks = []
        
        # 按面积降序排序，优先保留大的掩码
        for mask in sorted(masks, key=lambda x: x['area'], reverse=True):
            is_duplicate = False
            
            # 检查与已有掩码的重叠
            for existing_mask in filtered_masks:
                iou = self._calculate_iou(
                    mask['segmentation'], 
                    existing_mask['segmentation']
                )
                if iou > 0.5:  # IoU阈值
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                filtered_masks.append(mask)
                
        return filtered_masks
    
    def _calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个掩码的IoU（交并比）
        
        IoU = 交集 / 并集
        
        Args:
            mask1: 第一个掩码（布尔数组）
            mask2: 第二个掩码（布尔数组）
            
        Returns:
            IoU值，范围[0, 1]
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-8)
    
    def _classify_masks(
        self, 
        rgb_image: np.ndarray, 
        masks: List[Dict],
        spectral_indices: Dict[str, np.ndarray]
    ) -> List[Dict]:
        """
        对每个掩码进行语义分类
        
        使用CLIP进行零样本分类，并结合光谱指数进行校正
        
        Args:
            rgb_image: RGB图像
            masks: SAM生成的掩码列表
            spectral_indices: 光谱指数字典
            
        Returns:
            添加了分类信息的掩码列表
        """
        classified_masks = []
        
        # 准备CLIP的类别描述
        all_classes = []
        class_to_category = {}
        
        # 构建类别描述列表
        for category, classes in self.config.rs_classes.items():
            for cls in classes:
                # 添加"aerial view of"前缀以提高CLIP识别准确度
                class_desc = f"aerial view of {cls}"
                all_classes.append(class_desc)
                class_to_category[class_desc] = category
                
        # 对每个掩码进行分类
        for mask_data in tqdm(masks, desc="分类掩码"):
            # 提取掩码对应的图像区域
            bbox = mask_data['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # 裁剪出掩码区域的图像
            cropped_rgb = rgb_image[y1:y2, x1:x2]
            
            # 使用CLIP进行分类
            if cropped_rgb.size > 0:
                classification = self._classify_with_clip(cropped_rgb, all_classes)
                category = class_to_category.get(classification, 'unknown')
                
                # 计算掩码区域的光谱指数统计
                mask_indices = self._calculate_mask_indices(
                    mask_data['segmentation'], 
                    spectral_indices
                )
                
                # 基于光谱指数进行分类校正
                category = self._refine_classification(category, mask_indices)
                
                # 构建分类后的掩码数据
                classified_masks.append({
                    **mask_data,  # 保留原始掩码信息
                    'class': classification,  # CLIP分类结果
                    'category': category,  # 最终类别
                    'confidence': mask_data.get('predicted_iou', 0.5),  # 置信度
                    'spectral_features': mask_indices  # 光谱特征
                })
                
        return classified_masks
    
    def _classify_with_clip(self, image: np.ndarray, classes: List[str]) -> str:
        """
        使用CLIP模型进行零样本分类
        
        Args:
            image: 待分类的图像区域
            classes: 候选类别列表
            
        Returns:
            预测的类别名称
        """
        # 使用CLIP处理器预处理输入
        inputs = self.clip_processor(
            text=classes,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.config.device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)
            
        # 获取概率最高的类别
        predicted_idx = probs.argmax().item()
        return classes[predicted_idx]
    
    def _calculate_mask_indices(
        self, 
        mask: np.ndarray, 
        spectral_indices: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        计算掩码区域的光谱指数统计特征
        
        Args:
            mask: 布尔掩码
            spectral_indices: 光谱指数字典
            
        Returns:
            包含各光谱指数统计值的字典
        """
        mask_indices = {}
        
        for index_name, index_map in spectral_indices.items():
            # 提取掩码区域的指数值
            masked_values = index_map[mask]
            if len(masked_values) > 0:
                # 计算统计特征
                mask_indices[f'{index_name}_mean'] = np.mean(masked_values)
                mask_indices[f'{index_name}_std'] = np.std(masked_values)
                
        return mask_indices
    
    def _refine_classification(
        self, 
        category: str, 
        mask_indices: Dict[str, float]
    ) -> str:
        """
        基于光谱特征细化分类结果
        
        使用专家知识和光谱指数阈值来校正CLIP的分类结果
        
        Args:
            category: CLIP预测的类别
            mask_indices: 掩码区域的光谱指数统计
            
        Returns:
            校正后的类别
        """
        # 基于NDVI的规则
        if 'ndvi_mean' in mask_indices:
            ndvi = mask_indices['ndvi_mean']
            if ndvi > 0.6:
                # 高NDVI值表示茂密植被
                return 'vegetation'
            elif ndvi < 0:
                # 负NDVI值可能是水体或建筑
                return 'water' if mask_indices.get('ndwi_mean', 0) > 0.3 else 'urban'
                
        # 基于NDWI的规则
        if 'ndwi_mean' in mask_indices:
            ndwi = mask_indices['ndwi_mean']
            if ndwi > 0.3:
                # 高NDWI值表示水体
                return 'water'
                
        # 基于NDBI的规则
        if 'ndbi_mean' in mask_indices:
            ndbi = mask_indices['ndbi_mean']
            if ndbi > 0.1:
                # 正NDBI值可能表示建筑区域
                return 'urban'
                
        return category
    
    def _generate_segmentation_map(
        self, 
        shape: Tuple[int, int], 
        classified_masks: List[Dict]
    ) -> np.ndarray:
        """
        生成最终的语义分割图
        
        将所有分类后的掩码合并成一张完整的语义分割图
        
        Args:
            shape: 输出图像的形状 (H, W)
            classified_masks: 分类后的掩码列表
            
        Returns:
            语义分割图，每个像素值代表一个类别
        """
        # 类别到数值的映射
        category_colors = {
            'urban': 1,
            'vegetation': 2,
            'water': 3,
            'bare_land': 4,
            'unknown': 0
        }
        
        # 初始化分割图
        segmentation_map = np.zeros(shape, dtype=np.uint8)
        
        # 按面积排序，小的掩码后绘制（覆盖大的）
        sorted_masks = sorted(classified_masks, key=lambda x: x['area'])
        
        # 绘制每个掩码
        for mask_data in sorted_masks:
            category = mask_data.get('category', 'unknown')
            color = category_colors.get(category, 0)
            # 将掩码区域设置为对应的类别值
            segmentation_map[mask_data['segmentation']] = color
            
        return segmentation_map
    
    def _extract_metadata(self, image_path: str) -> Dict:
        """
        提取图像的元数据
        
        包括地理参考信息、图像尺寸、数据类型等
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            包含各种元数据的字典
        """
        with rasterio.open(image_path) as src:
            return {
                'crs': str(src.crs),  # 坐标参考系统
                'transform': src.transform.to_gdal(),  # 地理变换参数
                'bounds': src.bounds,  # 地理边界
                'shape': (src.height, src.width),  # 图像尺寸
                'count': src.count,  # 波段数
                'dtype': str(src.dtypes[0])  # 数据类型
            }