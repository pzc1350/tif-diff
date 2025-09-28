#!/usr/bin/env python
"""
演示遥感SAM模型增强功能（不需要外部依赖）

这个脚本展示了为SAM模型添加的遥感数据专门处理功能。
"""

def demo_sensor_configurations():
    """演示不同传感器的配置"""
    print("=== 遥感传感器专门配置 ===")
    print()
    
    # 模拟不同传感器类型的配置
    sensor_configs = {
        'landsat': {
            'spectral_indices': ['NDVI', 'NDWI', 'NDBI', 'SAVI', 'EVI', 'MNDWI'],
            'band_mapping': {
                1: 'coastal',    # Coastal/Aerosol
                2: 'blue',       # Blue
                3: 'green',      # Green
                4: 'red',        # Red
                5: 'nir',        # Near Infrared
                6: 'swir1',      # SWIR 1
                7: 'swir2'       # SWIR 2
            },
            'description': 'Landsat 8/9 多光谱传感器，中分辨率（30m）'
        },
        'sentinel2': {
            'spectral_indices': ['NDVI', 'NDWI', 'NDBI', 'SAVI', 'NDRE', 'MSI'],
            'band_mapping': {
                2: 'blue',         # B2 - Blue
                3: 'green',        # B3 - Green
                4: 'red',          # B4 - Red
                5: 'red_edge1',    # B5 - Red Edge 1
                8: 'nir',          # B8 - NIR
                11: 'swir1',       # B11 - SWIR 1
                12: 'swir2'        # B12 - SWIR 2
            },
            'description': 'Sentinel-2 多光谱传感器，高分辨率（10-20m）'
        },
        'modis': {
            'spectral_indices': ['NDVI', 'NDWI', 'NDBI', 'SAVI'],
            'band_mapping': {
                1: 'red',          # Band 1 - Red
                2: 'nir',          # Band 2 - NIR
                3: 'blue',         # Band 3 - Blue
                4: 'green',        # Band 4 - Green
            },
            'description': 'MODIS 低分辨率传感器（250-1000m），适合大尺度监测'
        }
    }
    
    for sensor, config in sensor_configs.items():
        print(f"传感器: {sensor.upper()}")
        print(f"  描述: {config['description']}")
        print(f"  支持的光谱指数: {', '.join(config['spectral_indices'])}")
        print(f"  主要波段: {', '.join(f'B{k}({v})' for k, v in list(config['band_mapping'].items())[:4])}")
        print()

def demo_viewing_directions():
    """演示不同观测方向的优化"""
    print("=== 观测方向自适应优化 ===")
    print()
    
    viewing_configs = {
        'nadir': {
            'pred_iou_thresh': 0.86,
            'stability_score_thresh': 0.92,
            'points_per_side': 32,
            'description': '垂直观测，标准参数设置'
        },
        'oblique': {
            'pred_iou_thresh': 0.82,  # 降低IoU阈值处理透视变形
            'stability_score_thresh': 0.88,
            'points_per_side': 36,   # 增加采样点
            'description': '斜视观测，调整参数处理透视变形'
        },
        'multi-angle': {
            'pred_iou_thresh': 0.80,
            'stability_score_thresh': 0.85,
            'points_per_side': 32,
            'crop_n_layers': 2,      # 增加裁剪层数
            'description': '多角度观测，更灵活的分割策略'
        }
    }
    
    for direction, config in viewing_configs.items():
        print(f"观测方向: {direction}")
        print(f"  描述: {config['description']}")
        print(f"  IoU阈值: {config['pred_iou_thresh']}")
        print(f"  稳定性阈值: {config['stability_score_thresh']}")
        print(f"  采样点数: {config['points_per_side']}")
        if 'crop_n_layers' in config:
            print(f"  裁剪层数: {config['crop_n_layers']}")
        print()

def demo_spectral_indices():
    """演示增强的光谱指数"""
    print("=== 增强的光谱指数计算 ===")
    print()
    
    indices = {
        'NDVI': {
            'formula': '(NIR - Red) / (NIR + Red)',
            'range': '[-1, 1]',
            'purpose': '植被密度和健康度',
            'traditional': True
        },
        'NDWI': {
            'formula': '(Green - NIR) / (Green + NIR)',
            'range': '[-1, 1]',
            'purpose': '水体识别',
            'traditional': True
        },
        'NDBI': {
            'formula': '(SWIR - NIR) / (SWIR + NIR)',
            'range': '[-1, 1]',
            'purpose': '建筑区域识别',
            'traditional': True
        },
        'SAVI': {
            'formula': '((NIR - Red) / (NIR + Red + L)) * (1 + L)',
            'range': '[-1.5, 1.5]',
            'purpose': '土壤调节植被指数',
            'traditional': True
        },
        'EVI': {
            'formula': '2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))',
            'range': '[-1, 1]',
            'purpose': '增强植被指数，减少大气影响',
            'traditional': False
        },
        'MNDWI': {
            'formula': '(Green - SWIR) / (Green + SWIR)',
            'range': '[-1, 1]',
            'purpose': '修正水体指数，减少建筑阴影',
            'traditional': False
        },
        'NDRE': {
            'formula': '(NIR - RedEdge) / (NIR + RedEdge)',
            'range': '[-1, 1]',
            'purpose': '红边植被指数，叶绿素敏感',
            'traditional': False
        },
        'MSI': {
            'formula': 'SWIR / NIR',
            'range': '[0, +∞]',
            'purpose': '水分胁迫指数',
            'traditional': False
        }
    }
    
    print("传统光谱指数:")
    for name, info in indices.items():
        if info['traditional']:
            print(f"  {name}: {info['purpose']}")
            print(f"    公式: {info['formula']}")
            print(f"    值域: {info['range']}")
            print()
    
    print("新增光谱指数:")
    for name, info in indices.items():
        if not info['traditional']:
            print(f"  {name}: {info['purpose']}")
            print(f"    公式: {info['formula']}")
            print(f"    值域: {info['range']}")
            print()

def demo_classification_refinement():
    """演示基于光谱特征的分类细化"""
    print("=== 智能分类细化规则 ===")
    print()
    
    classification_rules = [
        {
            'category': '茂密植被',
            'conditions': 'NDVI > 0.6 AND SAVI > 0.4',
            'example': 'NDVI=0.8, SAVI=0.6 → vegetation'
        },
        {
            'category': '农田',
            'conditions': '0.3 < NDVI ≤ 0.6 AND EVI > 0.3',
            'example': 'NDVI=0.4, EVI=0.35 → agriculture'
        },
        {
            'category': '水体',
            'conditions': 'NDWI > 0.3 OR MNDWI > 0.4',
            'example': 'NDWI=0.5, MNDWI=0.6 → water'
        },
        {
            'category': '城市建筑',
            'conditions': 'NDBI > 0.1 OR (NDVI < 0.1 AND NDWI < 0.1)',
            'example': 'NDBI=0.25, NDVI=0.05 → urban'
        },
        {
            'category': '裸地',
            'conditions': 'NDVI < 0.2 AND NDWI < 0.1 AND NDBI < 0.1',
            'example': 'NDVI=0.1, NDWI=0.05, NDBI=0.05 → bare_land'
        }
    ]
    
    print("多光谱指数决策树分类规则:")
    for i, rule in enumerate(classification_rules, 1):
        print(f"{i}. {rule['category']}")
        print(f"   条件: {rule['conditions']}")
        print(f"   示例: {rule['example']}")
        print()

def demo_export_formats():
    """演示多种导出格式"""
    print("=== 多种遥感标准格式导出 ===")
    print()
    
    formats = {
        'GeoTIFF': {
            'extension': '.tif',
            'description': '保持地理参考的栅格格式',
            'content': '语义分割图，每个像素值代表地物类别',
            'applications': ['GIS软件', '遥感图像处理软件', 'QGIS', 'ArcGIS']
        },
        'Shapefile': {
            'extension': '.shp',
            'description': '矢量多边形格式',
            'content': '分割掩码转换为多边形，包含属性信息',
            'applications': ['GIS分析', '空间数据库', '制图应用']
        },
        'JSON': {
            'extension': '.json',
            'description': '结构化数据格式',
            'content': '分类统计、光谱特征、元数据等',
            'applications': ['Web应用', '数据分析', 'API接口']
        }
    }
    
    for format_name, info in formats.items():
        print(f"格式: {format_name}")
        print(f"  文件扩展名: {info['extension']}")
        print(f"  描述: {info['description']}")
        print(f"  内容: {info['content']}")
        print(f"  应用: {', '.join(info['applications'])}")
        print()

def demo_enhanced_classes():
    """演示增强的遥感类别"""
    print("=== 增强的遥感类别定义 ===")
    print()
    
    enhanced_classes = {
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
    
    for category, classes in enhanced_classes.items():
        print(f"类别: {category.upper()}")
        print(f"  子类: {', '.join(classes[:5])}...")
        print(f"  总计: {len(classes)} 个具体类别")
        print()

def main():
    """主演示函数"""
    print("🛰️  SAM模型遥感数据专门处理功能演示")
    print("=" * 60)
    print()
    
    demo_sensor_configurations()
    demo_viewing_directions()
    demo_spectral_indices()
    demo_classification_refinement()
    demo_export_formats()
    demo_enhanced_classes()
    
    print("=== 功能总结 ===")
    print()
    print("✅ 1. 传感器特定优化:")
    print("   - Landsat, Sentinel-2, MODIS等主流传感器支持")
    print("   - 自动波段映射和预处理")
    print("   - 传感器特定的光谱指数组合")
    print()
    print("✅ 2. 观测方向自适应:")
    print("   - 垂直观测（nadir）标准配置")
    print("   - 斜视观测（oblique）透视变形处理")
    print("   - 多角度观测灵活分割策略")
    print()
    print("✅ 3. 增强光谱分析:")
    print("   - 新增EVI, MNDWI, NDRE, MSI等高级光谱指数")
    print("   - 光谱指导的图像增强")
    print("   - 基于专家知识的分类细化")
    print()
    print("✅ 4. 遥感标准格式:")
    print("   - GeoTIFF保持地理参考")
    print("   - Shapefile矢量格式导出")
    print("   - JSON结构化数据和统计信息")
    print()
    print("✅ 5. 智能分类系统:")
    print("   - 扩展的遥感类别词汇")
    print("   - 多光谱指数决策树")
    print("   - CLIP结果的光谱校正")
    print()
    print("🎯 这些增强功能使SAM模型更适合遥感数据的专业应用！")

if __name__ == "__main__":
    main()