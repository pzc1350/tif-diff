import unittest
import numpy as np
from pathlib import Path
import tempfile

from sam_rs_core import RemoteSensingConfig, SpectralIndicesCalculator

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
        self.assertTrue(np.all(ndwi <= 1))

class TestConfig(unittest.TestCase):
    """测试配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RemoteSensingConfig()
        
        self.assertEqual(config.sam_model_type, "vit_h")
        self.assertEqual(config.tile_size, 1024)
        self.assertEqual(config.overlap, 128)
        
    def test_custom_config(self):
        """测试自定义配置"""
        config = RemoteSensingConfig(
            tile_size=512,
            overlap=64,
            device="cpu"
        )
        
        self.assertEqual(config.tile_size, 512)
        self.assertEqual(config.overlap, 64)
        self.assertEqual(config.device, "cpu")

if __name__ == '__main__':
    unittest.main()