# TIF-Diff 快速使用指南

## 🚀 5分钟快速上手

### 1. 安装

```bash
# 克隆项目
git clone https://github.com/pzc1350/tif-diff.git
cd tif-diff

# 安装依赖
pip install -r requirements.txt

# 下载模型权重
python scripts/download_weights.py
```

### 2. 最简单的使用方式

```bash
# 一行命令处理图像
python quick_start.py your_image.tif
```

结果将保存在 `quick_output/` 目录下：
- `result.png` - 可视化结果
- `segmentation.tif` - 语义分割图
- `statistics.json` - 统计信息

### 3. Python代码使用

```python
from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM

# 基础配置
config = RemoteSensingConfig()
rs_sam = RemoteSensingSAM(config)

# 处理图像
result = rs_sam.process_remote_sensing_image("image.tif")

# 导出结果
rs_sam.export_results(result, "output", formats=['geotiff', 'shapefile', 'json'])
```

## 🛰️ 针对不同传感器的专门优化

### Landsat数据处理
```python
config = RemoteSensingConfig(
    sensor_type="landsat",           # Landsat专用优化
    viewing_direction="nadir",       # 垂直观测
    use_spectral_guidance=True,      # 启用光谱指导
    adaptive_sam_params=True         # 自适应SAM参数
)
```

### Sentinel-2数据处理
```python
config = RemoteSensingConfig(
    sensor_type="sentinel2",         # Sentinel-2专用优化
    tile_size=512,                   # 适合高分辨率的分块大小
    spectral_indices=['NDVI', 'NDRE', 'MSI']  # 利用红边波段
)
```

### 斜视观测数据处理
```python
config = RemoteSensingConfig(
    viewing_direction="oblique",     # 斜视观测优化
    adaptive_sam_params=True,        # 自动调整参数处理透视变形
)
```

## 📊 增强的光谱指数支持

系统支持多种高级光谱指数：

- **传统指数**: NDVI, NDWI, NDBI, SAVI
- **增强指数**: EVI, MNDWI, NDRE, MSI
- **自动计算**: 根据传感器类型自动选择最优指数组合

```python
from quick_start import quick_process

# 处理单张图像
results = quick_process("your_image.tif")

# 获取分割结果
segmentation = results['segmentation_map']
masks = results['masks']
```

## 📊 理解输出结果

### 语义分割图

每个像素的值代表一个类别：
- 0: 未知/背景
- 1: 城市区域（建筑、道路）
- 2: 植被（森林、草地、农田）
- 3: 水体（河流、湖泊）
- 4: 裸地（土壤、沙地）

### 掩码列表

每个掩码包含以下信息：
```python
{
    'segmentation': np.ndarray,  # 布尔掩码
    'category': str,             # 类别名称
    'confidence': float,         # 置信度
    'area': int,                # 面积（像素）
    'bbox': list,               # 边界框
    'spectral_features': dict   # 光谱特征
}
```

## 🎯 常见使用场景

### 场景1：批量处理

```python
from pathlib import Path
from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM

# 配置
config = RemoteSensingConfig()
rs_sam = RemoteSensingSAM(config)

# 批量处理
image_folder = Path("images/")
for img_path in image_folder.glob("*.tif"):
    results = rs_sam.process_remote_sensing_image(str(img_path))
    # 保存结果...
```

### 场景2：自定义类别

```python
config = RemoteSensingConfig(
    rs_classes={
        'building': ['house', 'apartment', 'office'],
        'agriculture': ['rice field', 'wheat field', 'orchard'],
        'forest': ['deciduous forest', 'coniferous forest'],
        'water': ['river', 'pond', 'reservoir']
    }
)
```

### 场景3：只计算特定光谱指数

```python
from sam_rs_core import SpectralIndicesCalculator
import rasterio

# 读取图像
with rasterio.open("image.tif") as src:
    nir = src.read(4)  # NIR波段
    red = src.read(3)  # Red波段

# 计算NDVI
ndvi = SpectralIndicesCalculator.calculate_ndvi(nir, red)
```

## 💡 性能优化建议

### 1. GPU加速
确保安装了CUDA版本的PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. 调整分块大小
对于内存有限的情况：
```python
config = RemoteSensingConfig(
    tile_size=512,  # 减小分块大小
    overlap=64      # 减小重叠区域
)
```

### 3. 降低分割精度以加快速度
```python
# 在配置中调整SAM参数
config = RemoteSensingConfig(
    min_mask_region_area=200  # 增大最小区域
)
```

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减小 `tile_size` 参数，或使用CPU模式

### Q: 处理速度太慢？
A: 1) 使用GPU 2) 减小图像分辨率 3) 调整分割参数

### Q: 分类不准确？
A: 1) 检查光谱波段映射 2) 自定义类别描述 3) 调整光谱指数阈值

## 📚 更多资源

- [完整API文档](./API.md)
- [功能详解](./FEATURES.md)
- [示例notebook](../examples/demo.ipynb)