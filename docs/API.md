# TIF-Diff API 参考文档

## 核心类

### RemoteSensingConfig

配置类，用于管理所有处理参数。

```python
class RemoteSensingConfig:
    """
    遥感处理配置类
    
    Parameters:
        sam_checkpoint (str): SAM模型权重路径
        sam_model_type (str): 模型类型 ['vit_h', 'vit_l', 'vit_b']
        device (str): 计算设备 ['cuda', 'cpu']
        spectral_indices (List[str]): 要计算的光谱指数
        tile_size (int): 分块大小（像素）
        overlap (int): 重叠区域大小（像素）
        min_mask_region_area (int): 最小掩码面积
        rs_classes (Dict[str, List[str]]): 类别定义
    """
```

**示例：**
```python
config = RemoteSensingConfig(
    sam_model_type="vit_h",
    device="cuda",
    tile_size=1024,
    spectral_indices=['NDVI', 'NDWI']
)
```

### RemoteSensingSAM

主处理类，提供遥感图像分割和分类功能。

```python
class RemoteSensingSAM:
    """
    遥感SAM处理类
    
    Methods:
        __init__(config): 初始化
        process_remote_sensing_image(image_path): 处理图像
    """
```

**主要方法：**

#### process_remote_sensing_image()
```python
def process_remote_sensing_image(self, image_path: str) -> Dict:
    """
    处理遥感图像
    
    Args:
        image_path (str): 图像文件路径
        
    Returns:
        Dict: {
            'segmentation_map': np.ndarray,  # 语义分割图
            'masks': List[Dict],             # 掩码列表
            'spectral_indices': Dict,        # 光谱指数
            'metadata': Dict                 # 元数据
        }
    """
```

### SpectralIndicesCalculator

光谱指数计算工具类。

```python
class SpectralIndicesCalculator:
    """
    光谱指数计算器（静态类）
    
    Methods:
        calculate_ndvi(nir, red): 计算NDVI
        calculate_ndwi(green, nir): 计算NDWI
        calculate_ndbi(swir, nir): 计算NDBI
        calculate_savi(nir, red, L=0.5): 计算SAVI
        calculate_all_indices(image_data): 计算所有指数
    """
```

**使用示例：**
```python
# 计算NDVI
ndvi = SpectralIndicesCalculator.calculate_ndvi(nir_band, red_band)

# 计算所有可用指数
indices = SpectralIndicesCalculator.calculate_all_indices({
    'nir': nir_band,
    'red': red_band,
    'green': green_band
})
```

### TemporalRemoteSensing

时序分析类。

```python
class TemporalRemoteSensing:
    """
    时序遥感分析
    
    Methods:
        analyze_temporal_series(image_paths, dates): 分析时间序列
        _detect_changes(temporal_results): 检测变化
        _analyze_trends(temporal_results): 分析趋势
    """
```

### RemoteSensingVisualizer

可视化类。

```python
class RemoteSensingVisualizer:
    """
    结果可视化
    
    Methods:
        visualize_results(results, save_path, show_indices): 综合可视化
        create_interactive_map(results, output_path): 创建交互地图
        export_results(results, output_dir, formats): 导出结果
    """
```

## 数据格式

### 输入格式

支持的图像格式：
- GeoTIFF (.tif, .tiff)
- HDF5 (.h5, .hdf5)
- NetCDF (.nc)

波段要求：
- 至少包含RGB三个波段
- 可选：NIR, SWIR等多光谱波段

### 输出格式

#### segmentation_map
```python
np.ndarray  # shape: (H, W), dtype: uint8
# 值含义：
# 0: unknown
# 1: urban
# 2: vegetation
# 3: water
# 4: bare_land
```

#### masks
```python
List[Dict]  # 每个字典包含：
{
    'segmentation': np.ndarray,      # 布尔掩码
    'bbox': [x1, y1, x2, y2],       # 边界框
    'area': int,                     # 面积
    'category': str,                 # 类别
    'confidence': float,             # 置信度
    'spectral_features': Dict[str, float]  # 光谱特征
}
```

## 错误处理

### 常见异常

```python
try:
    results = rs_sam.process_remote_sensing_image("image.tif")
except FileNotFoundError:
    print("图像文件不存在")
except ValueError as e:
    print(f"数据格式错误: {e}")
except RuntimeError as e:
    print(f"处理错误: {e}")
```

### 错误代码

| 错误类型 | 描述 | 解决方案 |
|---------|------|----------|
| FileNotFoundError | 文件不存在 | 检查路径 |
| ValueError | 无效的波段数据 | 检查波段映射 |
| RuntimeError | GPU内存不足 | 减小tile_size |
| KeyError | 缺少必需波段 | 检查数据完整性 |

## 最佳实践

### 1. 内存管理
```python
# 大图像处理
config = RemoteSensingConfig(
    tile_size=512,  # 使用较小的块
    overlap=64      # 适当的重叠
)
```

### 2. 批处理
```python
# 批量处理多个文件
from concurrent.futures import ProcessPoolExecutor

def process_file(file_path):
    return rs_sam.process_remote_sensing_image(file_path)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, file_paths))
```

### 3. 结果缓存
```python
import pickle

# 保存结果
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)

# 加载结果
with open('results.pkl', 'rb') as f:
    results = pickle.load(f)
```