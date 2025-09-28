#!/usr/bin/env python
"""
增强的遥感SAM处理示例

这个示例展示了如何使用增强的RemoteSensingSAM类来处理不同类型的遥感数据，
包括不同传感器类型、观测方向和光谱指导功能。
"""

import numpy as np
from pathlib import Path
from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM, SpectralIndicesCalculator

def example_landsat_processing():
    """Landsat数据处理示例"""
    print("=== Landsat遥感数据处理示例 ===")
    
    # 配置Landsat专用参数
    config = RemoteSensingConfig(
        sensor_type="landsat",           # Landsat传感器
        viewing_direction="nadir",       # 垂直观测
        use_spectral_guidance=True,      # 启用光谱指导
        adaptive_sam_params=True,        # 自适应SAM参数
        tile_size=1024,                  # 适合Landsat分辨率的分块大小
    )
    
    print(f"Landsat光谱指数: {config.spectral_indices}")
    print(f"自适应SAM参数: {config.adaptive_sam_params}")
    print(f"光谱指导: {config.use_spectral_guidance}")
    
    # 注意：实际使用时需要SAM模型权重文件
    # rs_sam = RemoteSensingSAM(config)
    # result = rs_sam.process_remote_sensing_image("landsat_image.tif")
    # rs_sam.export_results(result, "output/landsat_results")

def example_sentinel2_processing():
    """Sentinel-2数据处理示例"""
    print("\n=== Sentinel-2遥感数据处理示例 ===")
    
    # 配置Sentinel-2专用参数
    config = RemoteSensingConfig(
        sensor_type="sentinel2",         # Sentinel-2传感器
        viewing_direction="nadir",       # 垂直观测
        use_spectral_guidance=True,      # 启用光谱指导
        adaptive_sam_params=True,        # 自适应SAM参数
        tile_size=512,                   # 适合Sentinel-2高分辨率的分块大小
    )
    
    print(f"Sentinel-2光谱指数: {config.spectral_indices}")
    print(f"增强类别: {list(config.rs_classes.keys())}")

def example_oblique_viewing():
    """斜视观测数据处理示例"""
    print("\n=== 斜视观测遥感数据处理示例 ===")
    
    # 配置斜视观测参数
    config = RemoteSensingConfig(
        sensor_type="generic",
        viewing_direction="oblique",     # 斜视观测
        use_spectral_guidance=True,
        adaptive_sam_params=True,
    )
    
    print(f"观测方向: {config.viewing_direction}")
    print("斜视观测将自动调整SAM参数以处理透视变形")

def example_spectral_indices():
    """光谱指数计算示例"""
    print("\n=== 增强光谱指数计算示例 ===")
    
    # 创建模拟的多光谱数据
    height, width = 500, 500
    image_data = {
        'blue': np.random.rand(height, width) * 0.3,
        'green': np.random.rand(height, width) * 0.4,
        'red': np.random.rand(height, width) * 0.35,
        'nir': np.random.rand(height, width) * 0.6,
        'swir1': np.random.rand(height, width) * 0.25,
        'red_edge': np.random.rand(height, width) * 0.5
    }
    
    # 计算所有可用的光谱指数
    indices = SpectralIndicesCalculator.calculate_all_indices(image_data)
    
    print("计算出的光谱指数:")
    for name, index_map in indices.items():
        print(f"  {name.upper()}: 范围 [{np.min(index_map):.3f}, {np.max(index_map):.3f}], "
              f"均值 {np.mean(index_map):.3f}")

def example_multi_format_export():
    """多格式导出示例"""
    print("\n=== 多格式导出示例 ===")
    
    # 模拟处理结果
    results = {
        'segmentation_map': np.random.randint(0, 5, (100, 100)).astype(np.uint8),
        'masks': [
            {
                'segmentation': np.random.random((100, 100)) > 0.5,
                'category': 'vegetation',
                'class': 'forest',
                'confidence': 0.85,
                'area': 1500,
                'bbox': [10, 10, 50, 50],
                'spectral_features': {'ndvi_mean': 0.7, 'ndwi_mean': 0.1}
            },
            {
                'segmentation': np.random.random((100, 100)) > 0.7,
                'category': 'water',
                'class': 'river',
                'confidence': 0.92,
                'area': 800,
                'bbox': [60, 30, 80, 70],
                'spectral_features': {'ndvi_mean': -0.1, 'ndwi_mean': 0.6}
            }
        ],
        'spectral_indices': {
            'ndvi': np.random.rand(100, 100),
            'ndwi': np.random.rand(100, 100),
            'ndbi': np.random.rand(100, 100)
        },
        'metadata': {
            'crs': 'EPSG:4326',
            'transform': [0.1, 0.0, -120.0, 0.0, -0.1, 40.0],
            'bounds': [-120.0, 39.0, -119.0, 40.0],
            'shape': (100, 100),
            'count': 3,
            'dtype': 'float32'
        }
    }
    
    print("支持的导出格式:")
    print("  - GeoTIFF: 保持地理参考的栅格格式")
    print("  - Shapefile: 矢量多边形格式")
    print("  - JSON: 包含统计信息和元数据")
    
    # 实际导出（需要在有RemoteSensingSAM实例时使用）
    # rs_sam.export_results(results, "output/multi_format", 
    #                      formats=['geotiff', 'shapefile', 'json'])

def example_classification_refinement():
    """分类细化示例"""
    print("\n=== 基于光谱特征的分类细化示例 ===")
    
    # 模拟不同类型区域的光谱特征
    test_cases = [
        {
            'name': '茂密植被',
            'clip_result': 'vegetation',
            'indices': {'ndvi_mean': 0.8, 'ndwi_mean': 0.1, 'ndbi_mean': -0.1, 'savi_mean': 0.6}
        },
        {
            'name': '水体',
            'clip_result': 'unknown',
            'indices': {'ndvi_mean': -0.2, 'ndwi_mean': 0.5, 'ndbi_mean': -0.2, 'mndwi_mean': 0.6}
        },
        {
            'name': '城市建筑',
            'clip_result': 'urban',
            'indices': {'ndvi_mean': 0.05, 'ndwi_mean': -0.1, 'ndbi_mean': 0.3, 'savi_mean': 0.02}
        },
        {
            'name': '农田',
            'clip_result': 'vegetation',
            'indices': {'ndvi_mean': 0.4, 'ndwi_mean': 0.0, 'ndbi_mean': 0.0, 'evi_mean': 0.35}
        }
    ]
    
    # 模拟分类细化过程
    print("分类细化结果:")
    for case in test_cases:
        # 这里模拟RemoteSensingSAM的_refine_classification方法逻辑
        refined_category = simulate_refine_classification(case['clip_result'], case['indices'])
        print(f"  {case['name']}: CLIP={case['clip_result']} -> 细化后={refined_category}")

def simulate_refine_classification(category: str, mask_indices: dict) -> str:
    """模拟分类细化逻辑"""
    ndvi = mask_indices.get('ndvi_mean', 0)
    ndwi = mask_indices.get('ndwi_mean', 0)
    ndbi = mask_indices.get('ndbi_mean', 0)
    savi = mask_indices.get('savi_mean', 0)
    
    if ndvi > 0.6 and savi > 0.4:
        return 'vegetation'
    elif ndvi > 0.3 and ndvi <= 0.6:
        if 'evi_mean' in mask_indices and mask_indices['evi_mean'] > 0.3:
            return 'agriculture'
        else:
            return 'vegetation'
    elif ndwi > 0.3:
        return 'water'
    elif ndbi > 0.1 or (ndvi < 0.1 and ndwi < 0.1):
        return 'urban'
    elif ndvi < 0.2 and ndwi < 0.1 and ndbi < 0.1:
        return 'bare_land'
    
    return category

if __name__ == "__main__":
    print("增强的遥感SAM处理功能演示\n")
    
    # 运行各种示例
    example_landsat_processing()
    example_sentinel2_processing()
    example_oblique_viewing()
    example_spectral_indices()
    example_multi_format_export()
    example_classification_refinement()
    
    print("\n=== 总结 ===")
    print("增强功能包括:")
    print("1. 传感器特定的波段映射和光谱指数")
    print("2. 观测方向自适应的SAM参数优化")
    print("3. 光谱指导的图像增强")
    print("4. 基于专家知识的分类细化")
    print("5. 多种遥感标准格式导出")
    print("6. 增强的光谱指数计算（EVI, MNDWI, NDRE, MSI等）")
    print("\n这些功能使SAM模型更适合处理遥感数据的特殊需求！")