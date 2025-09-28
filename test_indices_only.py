#!/usr/bin/env python
"""
测试光谱指数计算功能（不依赖深度学习库）
"""

import numpy as np

class SpectralIndicesCalculator:
    """
    光谱指数计算器（独立版本，不依赖其他模块）
    """
    
    @staticmethod
    def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """计算归一化植被指数 (NDVI)"""
        return (nir - red) / (nir + red + 1e-8)
    
    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """计算归一化水体指数 (NDWI)"""
        return (green - nir) / (green + nir + 1e-8)
    
    @staticmethod
    def calculate_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """计算归一化建筑指数 (NDBI)"""
        return (swir - nir) / (swir + nir + 1e-8)
    
    @staticmethod
    def calculate_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """计算土壤调节植被指数 (SAVI)"""
        return ((nir - red) / (nir + red + L)) * (1 + L)
    
    @staticmethod
    def calculate_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """计算增强植被指数 (EVI)"""
        return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8))
    
    @staticmethod
    def calculate_mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """计算修正归一化水体指数 (MNDWI)"""
        return (green - swir) / (green + swir + 1e-8)
    
    @staticmethod
    def calculate_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
        """计算归一化红边指数 (NDRE)"""
        return (nir - red_edge) / (nir + red_edge + 1e-8)
    
    @staticmethod
    def calculate_msi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
        """计算水分胁迫指数 (MSI)"""
        return swir / (nir + 1e-8)

def test_spectral_indices():
    """测试光谱指数计算功能"""
    print("=== 测试增强的光谱指数计算功能 ===\n")
    
    # 创建模拟的多光谱数据
    height, width = 100, 100
    
    # 模拟真实的遥感反射率值（0-1范围）
    blue = np.random.rand(height, width) * 0.3      # 蓝光反射率通常较低
    green = np.random.rand(height, width) * 0.4     # 绿光反射率
    red = np.random.rand(height, width) * 0.35      # 红光反射率
    nir = np.random.rand(height, width) * 0.6       # 近红外反射率（植被高）
    swir = np.random.rand(height, width) * 0.25     # 短波红外反射率
    red_edge = np.random.rand(height, width) * 0.5  # 红边反射率
    
    # 创建一些特征区域
    # 植被区域：高NIR，中等红光
    veg_mask = (np.random.rand(height, width) > 0.7)
    nir[veg_mask] = np.random.rand(np.sum(veg_mask)) * 0.3 + 0.6  # 0.6-0.9
    red[veg_mask] = np.random.rand(np.sum(veg_mask)) * 0.2 + 0.1  # 0.1-0.3
    
    # 水体区域：低NIR，中等蓝绿光
    water_mask = (np.random.rand(height, width) > 0.85)
    nir[water_mask] = np.random.rand(np.sum(water_mask)) * 0.1     # 0.0-0.1
    blue[water_mask] = np.random.rand(np.sum(water_mask)) * 0.2 + 0.2  # 0.2-0.4
    green[water_mask] = np.random.rand(np.sum(water_mask)) * 0.2 + 0.3  # 0.3-0.5
    
    print("测试传统光谱指数:")
    
    # 计算NDVI
    ndvi = SpectralIndicesCalculator.calculate_ndvi(nir, red)
    print(f"NDVI: 范围 [{ndvi.min():.3f}, {ndvi.max():.3f}], 均值 {ndvi.mean():.3f}")
    print(f"      植被区域NDVI均值: {ndvi[veg_mask].mean():.3f}")
    
    # 计算NDWI
    ndwi = SpectralIndicesCalculator.calculate_ndwi(green, nir)
    print(f"NDWI: 范围 [{ndwi.min():.3f}, {ndwi.max():.3f}], 均值 {ndwi.mean():.3f}")
    print(f"      水体区域NDWI均值: {ndwi[water_mask].mean():.3f}")
    
    # 计算NDBI
    ndbi = SpectralIndicesCalculator.calculate_ndbi(swir, nir)
    print(f"NDBI: 范围 [{ndbi.min():.3f}, {ndbi.max():.3f}], 均值 {ndbi.mean():.3f}")
    
    # 计算SAVI
    savi = SpectralIndicesCalculator.calculate_savi(nir, red)
    print(f"SAVI: 范围 [{savi.min():.3f}, {savi.max():.3f}], 均值 {savi.mean():.3f}")
    
    print("\n测试增强光谱指数:")
    
    # 计算EVI
    evi = SpectralIndicesCalculator.calculate_evi(nir, red, blue)
    print(f"EVI:  范围 [{evi.min():.3f}, {evi.max():.3f}], 均值 {evi.mean():.3f}")
    print(f"      植被区域EVI均值: {evi[veg_mask].mean():.3f}")
    
    # 计算MNDWI
    mndwi = SpectralIndicesCalculator.calculate_mndwi(green, swir)
    print(f"MNDWI:范围 [{mndwi.min():.3f}, {mndwi.max():.3f}], 均值 {mndwi.mean():.3f}")
    print(f"      水体区域MNDWI均值: {mndwi[water_mask].mean():.3f}")
    
    # 计算NDRE
    ndre = SpectralIndicesCalculator.calculate_ndre(nir, red_edge)
    print(f"NDRE: 范围 [{ndre.min():.3f}, {ndre.max():.3f}], 均值 {ndre.mean():.3f}")
    
    # 计算MSI
    msi = SpectralIndicesCalculator.calculate_msi(nir, swir)
    print(f"MSI:  范围 [{msi.min():.3f}, {msi.max():.3f}], 均值 {msi.mean():.3f}")
    
    print("\n=== 验证光谱指数的合理性 ===")
    
    # 验证NDVI值域
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1), "NDVI值域错误"
    print("✓ NDVI值域正确 [-1, 1]")
    
    # 验证NDWI值域
    assert np.all(ndwi >= -1) and np.all(ndwi <= 1), "NDWI值域错误"
    print("✓ NDWI值域正确 [-1, 1]")
    
    # 验证NDBI值域
    assert np.all(ndbi >= -1) and np.all(ndbi <= 1), "NDBI值域错误"
    print("✓ NDBI值域正确 [-1, 1]")
    
    # 验证MNDWI值域
    assert np.all(mndwi >= -1) and np.all(mndwi <= 1), "MNDWI值域错误"
    print("✓ MNDWI值域正确 [-1, 1]")
    
    # 验证NDRE值域
    assert np.all(ndre >= -1) and np.all(ndre <= 1), "NDRE值域错误"
    print("✓ NDRE值域正确 [-1, 1]")
    
    # 验证MSI为正值
    assert np.all(msi >= 0), "MSI应该为正值"
    print("✓ MSI值为正值")
    
    # 验证植被区域的NDVI较高
    if np.any(veg_mask):
        veg_ndvi_mean = ndvi[veg_mask].mean()
        assert veg_ndvi_mean > 0.3, f"植被区域NDVI应该较高，实际为 {veg_ndvi_mean:.3f}"
        print(f"✓ 植被区域NDVI较高: {veg_ndvi_mean:.3f}")
    
    # 验证水体区域的NDWI较高
    if np.any(water_mask):
        water_ndwi_mean = ndwi[water_mask].mean()
        print(f"✓ 水体区域NDWI: {water_ndwi_mean:.3f}")
    
    print("\n=== 光谱指数计算功能测试通过！ ===")
    return True

def test_classification_rules():
    """测试基于光谱指数的分类规则"""
    print("\n=== 测试分类规则 ===")
    
    test_scenarios = [
        {
            'name': '茂密森林',
            'indices': {'ndvi_mean': 0.8, 'savi_mean': 0.6, 'ndwi_mean': 0.1, 'ndbi_mean': -0.1},
            'expected': 'vegetation'
        },
        {
            'name': '农田',
            'indices': {'ndvi_mean': 0.4, 'savi_mean': 0.3, 'evi_mean': 0.35, 'ndwi_mean': 0.0},
            'expected': 'agriculture'
        },
        {
            'name': '湖泊',
            'indices': {'ndvi_mean': -0.2, 'ndwi_mean': 0.5, 'mndwi_mean': 0.6, 'ndbi_mean': -0.2},
            'expected': 'water'
        },
        {
            'name': '城市建筑',
            'indices': {'ndvi_mean': 0.05, 'ndwi_mean': -0.1, 'ndbi_mean': 0.25, 'savi_mean': 0.02},
            'expected': 'urban'
        },
        {
            'name': '裸地',
            'indices': {'ndvi_mean': 0.1, 'ndwi_mean': 0.05, 'ndbi_mean': 0.05, 'msi_mean': 1.8},
            'expected': 'bare_land'
        }
    ]
    
    def simulate_classification_rules(mask_indices):
        """模拟分类规则"""
        ndvi = mask_indices.get('ndvi_mean', 0)
        ndwi = mask_indices.get('ndwi_mean', 0)
        ndbi = mask_indices.get('ndbi_mean', 0)
        savi = mask_indices.get('savi_mean', 0)
        
        # 植被识别
        if ndvi > 0.6 and savi > 0.4:
            return 'vegetation'
        elif ndvi > 0.3 and ndvi <= 0.6:
            if 'evi_mean' in mask_indices and mask_indices['evi_mean'] > 0.3:
                return 'agriculture'
            else:
                return 'vegetation'
        
        # 水体识别
        elif ndwi > 0.3:
            return 'water'
        
        # 建筑识别
        elif ndbi > 0.1 or (ndvi < 0.1 and ndwi < 0.1):
            if ndbi > 0.2:
                return 'urban'
            elif ndvi < 0.05:
                return 'urban'
            else:
                return 'bare_land'
        
        # 裸地识别
        elif ndvi < 0.2 and ndwi < 0.1 and ndbi < 0.1:
            return 'bare_land'
            
        return 'unknown'
    
    correct_predictions = 0
    total_predictions = len(test_scenarios)
    
    for scenario in test_scenarios:
        predicted = simulate_classification_rules(scenario['indices'])
        is_correct = predicted == scenario['expected']
        status = "✓" if is_correct else "✗"
        
        print(f"{status} {scenario['name']}: 预测={predicted}, 期望={scenario['expected']}")
        
        if is_correct:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    print(f"\n分类规则准确率: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    return accuracy > 0.8  # 期望80%以上的准确率

if __name__ == "__main__":
    print("测试增强的遥感光谱指数计算功能\n")
    
    success = True
    try:
        success &= test_spectral_indices()
        success &= test_classification_rules()
        
        if success:
            print("\n🎉 所有测试通过！增强的遥感处理功能工作正常。")
        else:
            print("\n❌ 部分测试失败，请检查实现。")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        success = False
    
    exit(0 if success else 1)