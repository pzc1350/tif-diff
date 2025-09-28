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
        sensor_type (str): 传感器类型，用于优化波段处理
        viewing_direction (str): 观测方向，影响SAM处理参数
        use_spectral_guidance (bool): 是否使用光谱指导SAM分割
        adaptive_sam_params (bool): 是否根据遥感数据特性自适应调整SAM参数
    """
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"  # 可选: vit_h, vit_l, vit_b
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 遥感特定参数
    spectral_indices: List[str] = field(default_factory=list)
    tile_size: int = 1024  # 分块大小
    overlap: int = 128  # 重叠区域
    min_mask_region_area: int = 100  # 最小掩码区域
    
    # 遥感专门化参数
    sensor_type: str = "generic"  # generic, landsat, sentinel2, modis, etc.
    viewing_direction: str = "nadir"  # nadir, oblique, multi-angle
    use_spectral_guidance: bool = True  # 使用光谱信息指导分割
    adaptive_sam_params: bool = True  # 自适应调整SAM参数
    
    # 类别定义
    rs_classes: Dict[str, List[str]] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if not self.spectral_indices:
            # 根据传感器类型设置默认光谱指数
            if self.sensor_type == "landsat":
                self.spectral_indices = ['NDVI', 'NDWI', 'NDBI', 'SAVI', 'EVI', 'MNDWI']
            elif self.sensor_type == "sentinel2":
                self.spectral_indices = ['NDVI', 'NDWI', 'NDBI', 'SAVI', 'NDRE', 'MSI']
            else:
                self.spectral_indices = ['NDVI', 'NDWI', 'NDBI', 'SAVI']
            
        if not self.rs_classes:
            # 增强的遥感类别定义，专门针对遥感应用优化
            self.rs_classes = {
                'urban': [
                    'residential building', 'commercial building', 'industrial building',
                    'road', 'highway', 'parking lot', 'concrete structure',
                    'urban area', 'built-up area', 'infrastructure'
                ],
                'vegetation': [
                    'dense forest', 'sparse forest', 'deciduous forest', 'coniferous forest',
                    'grassland', 'meadow', 'cropland', 'agricultural field',
                    'park', 'green space', 'vegetation cover'
                ],
                'water': [
                    'river', 'stream', 'lake', 'pond', 'reservoir',
                    'ocean', 'sea', 'coastal water', 'wetland', 'water body'
                ],
                'bare_land': [
                    'bare soil', 'exposed earth', 'sand', 'rock', 'quarry',
                    'desert', 'barren land', 'construction site', 'mining area'
                ],
                'agriculture': [
                    'crop field', 'farmland', 'agricultural area', 'cultivated land',
                    'greenhouse', 'orchard', 'vineyard', 'pasture'
                ],
                'natural': [
                    'mountain', 'hill', 'valley', 'cliff', 'natural terrain',
                    'geological formation', 'natural landscape'
                ]
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
    @staticmethod
    def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        计算增强植被指数 (EVI)
        
        EVI = 2.5 * ((NIR - Red) / (NIR + 6 * Red - 7.5 * Blue + 1))
        对大气和土壤背景更敏感，适合高密度植被区域
        
        Args:
            nir: 近红外波段数据
            red: 红光波段数据
            blue: 蓝光波段数据
            
        Returns:
            EVI数组
        """
        return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8))
    
    @staticmethod
    def calculate_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        计算修正归一化水体指数 (MNDWI)
        
        MNDWI = (Green - SWIR) / (Green + SWIR)
        对水体识别更准确，减少建筑阴影干扰
        
        Args:
            green: 绿光波段数据
            swir: 短波红外波段数据
            
        Returns:
            MNDWI数组
        """
        return (green - swir) / (green + swir + 1e-8)
    
    @staticmethod
    def calculate_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """
        计算归一化红边指数 (NDRE)
        
        NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        对植被叶绿素含量敏感，适用于Sentinel-2等传感器
        
        Args:
            nir: 近红外波段数据
            red_edge: 红边波段数据
            
        Returns:
            NDRE数组
        """
        return (nir - red_edge) / (nir + red_edge + 1e-8)
    
    @staticmethod
    def calculate_msi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """
        计算水分胁迫指数 (MSI)
        
        MSI = SWIR / NIR
        用于评估植被水分状况
        
        Args:
            nir: 近红外波段数据
            swir: 短波红外波段数据
            
        Returns:
            MSI数组
        """
        return swir / (nir + 1e-8)
    
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
        swir_band = image_data.get('swir', image_data.get('swir1', None))
        if swir_band is not None and 'nir' in image_data:
            indices['ndbi'] = SpectralIndicesCalculator.calculate_ndbi(
                swir_band, image_data['nir']
            )
            
        # 计算SAVI
        if 'nir' in image_data and 'red' in image_data:
            indices['savi'] = SpectralIndicesCalculator.calculate_savi(
                image_data['nir'], image_data['red']
            )
            
        # 计算EVI
        if 'nir' in image_data and 'red' in image_data and 'blue' in image_data:
            indices['evi'] = SpectralIndicesCalculator.calculate_evi(
                image_data['nir'], image_data['red'], image_data['blue']
            )
            
        # 计算MNDWI
        if 'green' in image_data and swir_band is not None:
            indices['mndwi'] = SpectralIndicesCalculator.calculate_mndwi(
                image_data['green'], swir_band
            )
            
        # 计算NDRE (适用于Sentinel-2等具有红边波段的传感器)
        if 'nir' in image_data and 'red_edge' in image_data:
            indices['ndre'] = SpectralIndicesCalculator.calculate_ndre(
                image_data['nir'], image_data['red_edge']
            )
            
        # 计算MSI
        if 'nir' in image_data and swir_band is not None:
            indices['msi'] = SpectralIndicesCalculator.calculate_msi(
                image_data['nir'], swir_band
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
        
        # 根据遥感数据特性配置SAM参数
        sam_params = self._get_rs_optimized_sam_params()
        
        # 配置自动掩码生成器，针对遥感数据优化
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            **sam_params
        )
        
        # 初始化CLIP模型用于语义理解
        print("加载CLIP模型...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(config.device)
        
    def _get_rs_optimized_sam_params(self) -> Dict:
        """
        获取针对遥感数据优化的SAM参数
        
        根据传感器类型、观测方向等因素调整SAM参数
        
        Returns:
            优化后的SAM参数字典
        """
        base_params = {
            'points_per_side': 32,  # 每边采样点数
            'pred_iou_thresh': 0.86,  # 预测IoU阈值
            'stability_score_thresh': 0.92,  # 稳定性分数阈值
            'crop_n_layers': 1,  # 裁剪层数
            'crop_n_points_downscale_factor': 2,  # 裁剪点下采样因子
            'min_mask_region_area': self.config.min_mask_region_area
        }
        
        if not self.config.adaptive_sam_params:
            return base_params
            
        # 根据观测方向调整参数
        if self.config.viewing_direction == "oblique":
            # 斜视观测时，目标可能有透视变形，降低IoU阈值
            base_params['pred_iou_thresh'] = 0.82
            base_params['stability_score_thresh'] = 0.88
            base_params['points_per_side'] = 36  # 增加采样点
            
        elif self.config.viewing_direction == "multi-angle":
            # 多角度观测时，需要更灵活的分割
            base_params['pred_iou_thresh'] = 0.80
            base_params['stability_score_thresh'] = 0.85
            base_params['crop_n_layers'] = 2  # 增加裁剪层数
            
        # 根据传感器类型调整参数
        if self.config.sensor_type == "modis":
            # MODIS分辨率较低，需要调整最小区域面积
            base_params['min_mask_region_area'] = max(500, self.config.min_mask_region_area)
            base_params['points_per_side'] = 24  # 减少采样点
            
        elif self.config.sensor_type in ["landsat", "sentinel2"]:
            # 中分辨率传感器的标准设置
            base_params['points_per_side'] = 32
            base_params['min_mask_region_area'] = max(100, self.config.min_mask_region_area)
            
        elif "high_resolution" in self.config.sensor_type:
            # 高分辨率传感器，可以检测更小的目标
            base_params['min_mask_region_area'] = max(50, self.config.min_mask_region_area)
            base_params['points_per_side'] = 40  # 增加采样点以捕获细节
            
        return base_params
        
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
            
            # 根据传感器类型定义波段映射
            band_mapping = self._get_sensor_band_mapping(src.count)
            
            # 读取每个波段
            for i in range(1, src.count + 1):
                band_data = src.read(i)
                
                # 数据预处理：处理NoData值和异常值
                band_data = self._preprocess_band_data(band_data, src.nodata)
                
                if i in band_mapping:
                    bands[band_mapping[i]] = band_data
                else:
                    bands[f'band_{i}'] = band_data
                    
            # 保存地理参考信息，后续导出时使用
            self.transform = src.transform
            self.crs = src.crs
            
        return bands
    
    def _get_sensor_band_mapping(self, band_count: int) -> Dict[int, str]:
        """
        根据传感器类型获取波段映射
        
        Args:
            band_count: 波段总数
            
        Returns:
            波段编号到波段名称的映射字典
        """
        if self.config.sensor_type == "landsat":
            # Landsat 8/9 波段映射
            if band_count >= 7:
                return {
                    1: 'coastal',    # Coastal/Aerosol
                    2: 'blue',       # Blue
                    3: 'green',      # Green
                    4: 'red',        # Red
                    5: 'nir',        # Near Infrared
                    6: 'swir1',      # SWIR 1
                    7: 'swir2'       # SWIR 2
                }
            else:
                # Landsat 5/7 波段映射
                return {
                    1: 'blue',
                    2: 'green',
                    3: 'red',
                    4: 'nir',
                    5: 'swir1',
                    7: 'swir2'
                }
                
        elif self.config.sensor_type == "sentinel2":
            # Sentinel-2 波段映射（选择主要波段）
            return {
                1: 'coastal',      # B1 - Coastal aerosol
                2: 'blue',         # B2 - Blue
                3: 'green',        # B3 - Green
                4: 'red',          # B4 - Red
                5: 'red_edge1',    # B5 - Red Edge 1
                6: 'red_edge2',    # B6 - Red Edge 2
                7: 'red_edge3',    # B7 - Red Edge 3
                8: 'nir',          # B8 - NIR
                9: 'nir_narrow',   # B8A - NIR Narrow
                10: 'water_vapor', # B9 - Water vapor
                11: 'swir1',       # B11 - SWIR 1
                12: 'swir2'        # B12 - SWIR 2
            }
            
        elif self.config.sensor_type == "modis":
            # MODIS波段映射（部分常用波段）
            return {
                1: 'red',          # Band 1 - Red
                2: 'nir',          # Band 2 - NIR
                3: 'blue',         # Band 3 - Blue
                4: 'green',        # Band 4 - Green
                5: 'swir1',        # Band 5 - SWIR 1
                6: 'swir2',        # Band 6 - SWIR 2
                7: 'swir3'         # Band 7 - SWIR 3
            }
            
        else:
            # 通用映射
            mapping = {
                1: 'blue',
                2: 'green', 
                3: 'red',
                4: 'nir'
            }
            if band_count >= 5:
                mapping[5] = 'swir1'
            if band_count >= 6:
                mapping[6] = 'swir2'
            if band_count >= 7:
                mapping[7] = 'red_edge'
                
            return mapping
    
    def _preprocess_band_data(self, band_data: np.ndarray, nodata_value) -> np.ndarray:
        """
        预处理波段数据
        
        处理NoData值、异常值，并进行必要的数据类型转换
        
        Args:
            band_data: 原始波段数据
            nodata_value: NoData值
            
        Returns:
            预处理后的波段数据
        """
        # 处理NoData值
        if nodata_value is not None:
            band_data = np.where(band_data == nodata_value, 0, band_data)
            
        # 处理异常值（通常是由于传感器故障或大气影响）
        # 使用99.5%分位数作为上限
        upper_limit = np.percentile(band_data[band_data > 0], 99.5)
        band_data = np.clip(band_data, 0, upper_limit)
        
        # 数据归一化（如果数据是整型且范围很大）
        if band_data.dtype in [np.uint16, np.int16] and band_data.max() > 1:
            # 假设是DN值，进行基本的归一化
            if band_data.max() > 10000:  # 可能是Landsat等DN值
                band_data = band_data.astype(np.float32) / 10000.0
            else:  # 可能是Sentinel-2等反射率值
                band_data = band_data.astype(np.float32) / 10000.0
        elif band_data.dtype == np.uint8:
            # 8位数据归一化到[0,1]
            band_data = band_data.astype(np.float32) / 255.0
            
        return band_data
    
    def _create_enhanced_rgb(
        self, 
        image_data: Dict[str, np.ndarray], 
        spectral_indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        创建增强的RGB图像用于SAM处理
        
        将多光谱数据转换为RGB格式，并进行增强以提高分割效果。
        可选地使用光谱信息进行增强。
        
        Args:
            image_data: 包含各波段数据的字典
            spectral_indices: 计算出的光谱指数
            
        Returns:
            增强后的RGB图像 (H, W, 3)
        """
        # 获取RGB波段（优先使用标准波段名称）
        r = image_data.get('red', image_data.get('band_3', image_data.get('band_4', None)))
        g = image_data.get('green', image_data.get('band_2', image_data.get('band_3', None)))
        b = image_data.get('blue', image_data.get('band_1', image_data.get('band_2', None)))
        
        if r is None or g is None or b is None:
            # 尝试使用近红外创建伪彩色合成
            nir = image_data.get('nir', image_data.get('band_4', image_data.get('band_5', None)))
            if nir is not None:
                r = nir  # NIR作为红色通道
                g = image_data.get('red', image_data.get('band_3', r))  # Red作为绿色通道
                b = image_data.get('green', image_data.get('band_2', g))  # Green作为蓝色通道
            else:
                raise ValueError("无法找到足够的波段创建RGB图像")
            
        # 堆叠为RGB图像
        rgb = np.stack([r, g, b], axis=-1)
        
        # 使用光谱指导增强（如果启用）
        if self.config.use_spectral_guidance and spectral_indices:
            rgb = self._apply_spectral_guidance(rgb, spectral_indices)
        
        # 增强对比度以提高分割效果
        rgb = self._enhance_contrast(rgb)
        
        # 转换为uint8格式
        rgb = (rgb * 255).astype(np.uint8)
        
        return rgb
    
    def _apply_spectral_guidance(
        self, 
        rgb: np.ndarray, 
        spectral_indices: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        使用光谱指数信息增强RGB图像
        
        根据光谱指数调整RGB各通道，突出不同地物特征
        
        Args:
            rgb: 原始RGB图像
            spectral_indices: 光谱指数字典
            
        Returns:
            光谱增强后的RGB图像
        """
        enhanced_rgb = rgb.copy()
        
        # 使用NDVI增强植被
        if 'ndvi' in spectral_indices:
            ndvi = spectral_indices['ndvi']
            vegetation_mask = ndvi > 0.3
            # 增强绿色通道以突出植被
            enhanced_rgb[:, :, 1] = np.where(
                vegetation_mask,
                np.clip(enhanced_rgb[:, :, 1] * 1.2, 0, 1),
                enhanced_rgb[:, :, 1]
            )
            
        # 使用NDWI增强水体
        if 'ndwi' in spectral_indices:
            ndwi = spectral_indices['ndwi']
            water_mask = ndwi > 0.1
            # 增强蓝色通道以突出水体
            enhanced_rgb[:, :, 2] = np.where(
                water_mask,
                np.clip(enhanced_rgb[:, :, 2] * 1.3, 0, 1),
                enhanced_rgb[:, :, 2]
            )
            
        # 使用NDBI增强建筑
        if 'ndbi' in spectral_indices:
            ndbi = spectral_indices['ndbi']
            urban_mask = ndbi > 0.05
            # 调整RGB平衡以突出建筑区域
            enhanced_rgb[:, :, 0] = np.where(
                urban_mask,
                np.clip(enhanced_rgb[:, :, 0] * 1.1, 0, 1),
                enhanced_rgb[:, :, 0]
            )
            
        return enhanced_rgb
    
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
        
        使用专家知识和光谱指数阈值来校正CLIP的分类结果，
        结合多个光谱指数进行综合判断
        
        Args:
            category: CLIP预测的类别
            mask_indices: 掩码区域的光谱指数统计
            
        Returns:
            校正后的类别
        """
        # 基于多光谱指数的决策树分类
        
        # 获取关键光谱指数
        ndvi = mask_indices.get('ndvi_mean', 0)
        ndwi = mask_indices.get('ndwi_mean', 0)
        ndbi = mask_indices.get('ndbi_mean', 0)
        savi = mask_indices.get('savi_mean', 0)
        
        # 植被识别规则
        if ndvi > 0.6 and savi > 0.4:
            # 高NDVI和SAVI表示茂密植被
            if ndvi > 0.8:
                return 'vegetation'  # 茂密植被
            else:
                return 'vegetation'  # 中等植被
                
        elif ndvi > 0.3 and ndvi <= 0.6:
            # 中等NDVI值，可能是稀疏植被或农田
            if 'evi_mean' in mask_indices and mask_indices['evi_mean'] > 0.3:
                return 'agriculture'  # 农田
            else:
                return 'vegetation'  # 稀疏植被
                
        # 水体识别规则
        elif ndwi > 0.3:
            # 高NDWI值表示水体
            if 'mndwi_mean' in mask_indices and mask_indices['mndwi_mean'] > 0.4:
                return 'water'  # 确定的水体
            elif ndvi < 0:
                return 'water'  # 负NDVI也支持是水体
            else:
                return 'water'  # 可能的水体
                
        # 建筑/城市区域识别规则
        elif ndbi > 0.1 or (ndvi < 0.1 and ndwi < 0.1):
            # 正NDBI或低植被低水分指数表示建筑区域
            if ndbi > 0.2:
                return 'urban'  # 高密度建成区
            elif ndvi < 0.05:
                return 'urban'  # 低植被覆盖的城市区域
            else:
                return 'bare_land'  # 可能是裸地
                
        # 裸地识别规则
        elif ndvi < 0.2 and ndwi < 0.1 and ndbi < 0.1:
            # 低植被、低水分、低建筑指数表示裸地
            if 'msi_mean' in mask_indices:
                msi = mask_indices['msi_mean']
                if msi > 1.5:
                    return 'bare_land'  # 干燥裸地
                else:
                    return 'bare_land'  # 湿润裸地
            else:
                return 'bare_land'
                
        # 农业区域特殊处理
        elif 0.2 <= ndvi <= 0.6 and 'evi_mean' in mask_indices:
            evi = mask_indices['evi_mean']
            if evi > 0.25:
                return 'agriculture'  # 农田
                
        # 如果无法通过光谱指数确定，则使用CLIP结果
        # 但进行一些基本的合理性检查
        if category == 'vegetation' and ndvi < 0.1:
            # CLIP认为是植被但NDVI很低，可能误分类
            return 'bare_land'
        elif category == 'water' and ndwi < 0:
            # CLIP认为是水体但NDWI为负，可能误分类
            if ndbi > 0.1:
                return 'urban'
            else:
                return 'bare_land'
        elif category == 'urban' and ndvi > 0.5:
            # CLIP认为是城市但NDVI很高，可能是公园绿地
            return 'vegetation'
            
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
    
    def export_results(
        self,
        results: Dict,
        output_dir: str,
        formats: List[str] = ['geotiff', 'shapefile', 'json']
    ) -> None:
        """
        导出处理结果为多种遥感格式
        
        Args:
            results: process_remote_sensing_image返回的结果字典
            output_dir: 输出目录
            formats: 导出格式列表
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        segmentation_map = results['segmentation_map']
        masks = results['masks']
        metadata = results['metadata']
        
        # 导出GeoTIFF格式
        if 'geotiff' in formats:
            self._export_geotiff(
                segmentation_map,
                output_path / 'segmentation.tif',
                metadata
            )
            
        # 导出Shapefile格式
        if 'shapefile' in formats:
            self._export_shapefile(
                masks,
                output_path / 'segments.shp',
                metadata
            )
            
        # 导出JSON格式
        if 'json' in formats:
            self._export_json(
                results,
                output_path / 'results.json'
            )
            
        print(f"结果已导出到: {output_dir}")
    
    def _export_geotiff(
        self,
        segmentation_map: np.ndarray,
        output_path: Path,
        metadata: Dict
    ) -> None:
        """导出为GeoTIFF格式"""
        try:
            import rasterio
            from rasterio.transform import from_gdal
            
            # 重建Transform对象
            if hasattr(self, 'transform'):
                transform = self.transform
            else:
                transform = from_gdal(*metadata['transform'])
                
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=segmentation_map.shape[0],
                width=segmentation_map.shape[1],
                count=1,
                dtype=segmentation_map.dtype,
                crs=self.crs if hasattr(self, 'crs') else metadata.get('crs'),
                transform=transform
            ) as dst:
                dst.write(segmentation_map, 1)
                
        except ImportError:
            print("警告: 无法导出GeoTIFF格式，请安装rasterio")
    
    def _export_shapefile(
        self,
        masks: List[Dict],
        output_path: Path,
        metadata: Dict
    ) -> None:
        """导出为Shapefile格式"""
        try:
            import geopandas as gpd
            from shapely.geometry import Polygon
            from rasterio.features import shapes
            from rasterio.transform import from_gdal
            import pandas as pd
            
            # 准备数据
            geometries = []
            attributes = []
            
            # 重建Transform对象
            if hasattr(self, 'transform'):
                transform = self.transform
            else:
                transform = from_gdal(*metadata['transform'])
            
            for i, mask_data in enumerate(masks):
                # 将掩码转换为多边形
                mask = mask_data['segmentation'].astype(np.uint8)
                for geom, value in shapes(mask, transform=transform):
                    if value > 0:  # 只处理非零值
                        geometries.append(Polygon(geom['coordinates'][0]))
                        attributes.append({
                            'id': i,
                            'category': mask_data.get('category', 'unknown'),
                            'class': mask_data.get('class', 'unknown'),
                            'confidence': mask_data.get('confidence', 0.0),
                            'area': mask_data.get('area', 0)
                        })
            
            if geometries:
                # 创建GeoDataFrame
                gdf = gpd.GeoDataFrame(attributes, geometry=geometries)
                if hasattr(self, 'crs'):
                    gdf.crs = self.crs
                    
                # 保存为Shapefile
                gdf.to_file(output_path)
                
        except ImportError:
            print("警告: 无法导出Shapefile格式，请安装geopandas")
    
    def _export_json(self, results: Dict, output_path: Path) -> None:
        """导出为JSON格式"""
        import json
        
        # 准备可序列化的数据
        export_data = {
            'metadata': results['metadata'],
            'spectral_indices_stats': {},
            'classification_summary': {},
            'masks': []
        }
        
        # 光谱指数统计
        for name, index_map in results['spectral_indices'].items():
            export_data['spectral_indices_stats'][name] = {
                'mean': float(np.mean(index_map)),
                'std': float(np.std(index_map)),
                'min': float(np.min(index_map)),
                'max': float(np.max(index_map))
            }
            
        # 分类统计
        categories = {}
        for mask in results['masks']:
            category = mask.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
            
        export_data['classification_summary'] = categories
        
        # 掩码信息（不包含大数组）
        for mask in results['masks']:
            mask_info = {
                'category': mask.get('category', 'unknown'),
                'class': mask.get('class', 'unknown'),
                'confidence': mask.get('confidence', 0.0),
                'area': mask.get('area', 0),
                'bbox': mask.get('bbox', []),
                'spectral_features': mask.get('spectral_features', {})
            }
            export_data['masks'].append(mask_info)
        
        # 保存JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    
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