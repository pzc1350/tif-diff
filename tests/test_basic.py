import unittest
import numpy as np
from pathlib import Path
import tempfile

from sam_rs_core import RemoteSensingConfig, SpectralIndicesCalculator, RemoteSensingSAM

class TestSpectralIndices(unittest.TestCase):
    """测试光谱指数计算"""
    
    def setUp(self):
        # 创建模拟数据
        self.nir = np.random.rand(100, 100)
        self.red = np.random.rand(100, 100)
        self.green = np.random.rand(100, 100)
        
    def test_ndvi_calculation(self):
        """测试NDVI计算"""
        ndvi = SpectralIndicesCalculator.calculate_ndvi(self.nir, self.red)
        
        # 检查值域
        self.assertTrue(np.all(ndvi >= -1))
        self.assertTrue(np.all(ndvi <= 1))
        
        # 检查特殊情况
        nir_zero = np.zeros((10, 10))
        red_zero = np.zeros((10, 10))
        ndvi_zero = SpectralIndicesCalculator.calculate_ndvi(nir_zero, red_zero)
        self.assertTrue(np.all(ndvi_zero == 0))
        
    def test_ndwi_calculation(self):
        """测试NDWI计算"""
        ndwi = SpectralIndicesCalculator.calculate_ndwi(self.green, self.nir)
        
        # 检查值域
        self.assertTrue(np.all(ndwi >= -1))
    def test_enhanced_spectral_indices(self):
        """测试增强的光谱指数计算"""
        # 测试EVI
        evi = SpectralIndicesCalculator.calculate_evi(self.nir, self.red, np.random.rand(100, 100))
        self.assertIsInstance(evi, np.ndarray)
        
        # 测试MNDWI
        swir = np.random.rand(100, 100)
        mndwi = SpectralIndicesCalculator.calculate_mndwi(self.green, swir)
        self.assertTrue(np.all(mndwi >= -1))
        self.assertTrue(np.all(mndwi <= 1))
        
        # 测试NDRE
        red_edge = np.random.rand(100, 100)
        ndre = SpectralIndicesCalculator.calculate_ndre(self.nir, red_edge)
        self.assertTrue(np.all(ndre >= -1))
        self.assertTrue(np.all(ndre <= 1))
        
        # 测试MSI
        msi = SpectralIndicesCalculator.calculate_msi(self.nir, swir)
        self.assertTrue(np.all(msi >= 0))  # MSI应该是正值

class TestConfig(unittest.TestCase):
    """测试配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RemoteSensingConfig()
        
        self.assertEqual(config.sam_model_type, "vit_h")
        self.assertEqual(config.tile_size, 1024)
        self.assertEqual(config.overlap, 128)
        self.assertEqual(config.sensor_type, "generic")
        self.assertEqual(config.viewing_direction, "nadir")
        self.assertTrue(config.use_spectral_guidance)
        self.assertTrue(config.adaptive_sam_params)
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = RemoteSensingConfig(
            tile_size=512,
            overlap=64,
            device="cpu",
            sensor_type="landsat",
            viewing_direction="oblique"
        )
        
        self.assertEqual(config.tile_size, 512)
        self.assertEqual(config.overlap, 64)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.sensor_type, "landsat")
        self.assertEqual(config.viewing_direction, "oblique")
        
    def test_sensor_specific_indices(self):
        """测试传感器特定的光谱指数配置"""
        # Landsat配置
        landsat_config = RemoteSensingConfig(sensor_type="landsat")
        self.assertIn('EVI', landsat_config.spectral_indices)
        self.assertIn('MNDWI', landsat_config.spectral_indices)
        
        # Sentinel-2配置
        s2_config = RemoteSensingConfig(sensor_type="sentinel2")
        self.assertIn('NDRE', s2_config.spectral_indices)
        self.assertIn('MSI', s2_config.spectral_indices)

class TestRemoteSensingSAM(unittest.TestCase):
    """测试RemoteSensingSAM类"""
    
    def test_sam_params_optimization(self):
        """测试SAM参数优化"""
        # 测试斜视观测配置
        config = RemoteSensingConfig(
            viewing_direction="oblique",
            adaptive_sam_params=True
        )
        
        # 这里我们不实际初始化SAM模型（因为需要权重文件）
        # 只测试参数获取方法
        rs_sam = type('MockRemoteSensingSAM', (), {
            'config': config,
            '_get_rs_optimized_sam_params': RemoteSensingSAM._get_rs_optimized_sam_params
        })()
        
        params = rs_sam._get_rs_optimized_sam_params()
        
        # 斜视观测应该有特殊的参数设置
        self.assertEqual(params['pred_iou_thresh'], 0.82)
        self.assertEqual(params['stability_score_thresh'], 0.88)
        self.assertEqual(params['points_per_side'], 36)
        
    def test_band_mapping(self):
        """测试波段映射功能"""
        config = RemoteSensingConfig(sensor_type="landsat")
        
        # 模拟RemoteSensingSAM的波段映射方法
        rs_sam = type('MockRemoteSensingSAM', (), {
            'config': config,
            '_get_sensor_band_mapping': RemoteSensingSAM._get_sensor_band_mapping
        })()
        
        # 测试Landsat 8波段映射
        mapping = rs_sam._get_sensor_band_mapping(7)
        self.assertEqual(mapping[1], 'coastal')
        self.assertEqual(mapping[2], 'blue')
        self.assertEqual(mapping[5], 'nir')
        
        # 测试Sentinel-2波段映射
        config.sensor_type = "sentinel2"
        mapping = rs_sam._get_sensor_band_mapping(12)
        self.assertEqual(mapping[2], 'blue')
        self.assertEqual(mapping[8], 'nir')
        self.assertEqual(mapping[11], 'swir1')

if __name__ == '__main__':
    unittest.main()