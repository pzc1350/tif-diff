# TIF-Diff: 先进的遥感图像智能解译系统

基于 Segment Anything Model (SAM) 和 CLIP 的零样本遥感图像语义分割系统，无需训练即可使用。

## ✨ 特性

- 🚀 **零样本学习**：使用 SAM + CLIP，无需训练数据
- 🛰️ **多光谱支持**：自动计算 NDVI、NDWI、NDBI 等光谱指数
- 🗺️ **大图像处理**：自动分块处理超大遥感图像
- 📈 **时序分析**：支持多时相变化检测和趋势分析
- 📊 **丰富的可视化**：生成分割图、统计图表和交互式地图
- 📁 **多格式导出**：支持 GeoTIFF、Shapefile、JSON 等格式

## 🚀 快速开始

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/pzc1350/tif-diff.git
cd tif-diff

# 安装依赖
pip install -r requirements.txt

# 下载 SAM 模型权重
python scripts/download_weights.py
```

### 快速使用

```bash
# 处理单张遥感图像
python quick_start.py your_image.tif

# 使用完整功能
python main.py --input your_image.tif --output results --visualize

# 批量处理
python main.py --input image_folder/ --output results --mode batch
```

### Python API

```python
from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM

# 初始化
config = RemoteSensingConfig()
rs_sam = RemoteSensingSAM(config)

# 处理图像
results = rs_sam.process_remote_sensing_image("image.tif")

# 获取分割结果
segmentation_map = results['segmentation_map']
masks = results['masks']
```

## 📂 项目结构

```
tif-diff/
├── sam_rs_core.py          # 核心SAM遥感处理模块
├── temporal_analysis.py    # 时序分析模块
├── visualization.py        # 可视化模块
├── quick_start.py         # 快速开始脚本
├── main.py               # 主程序入口
├── requirements.txt      # 依赖列表
├── scripts/             # 工具脚本
│   └── download_weights.py
├── configs/             # 配置文件
│   └── default_config.yaml
├── examples/            # 示例
│   └── demo.ipynb
└── tests/              # 测试
```

## 🛠️ 功能详解

### 1. 语义分割
- 自动识别：建筑、道路、植被、水体、裸土等
- 基于 SAM 的精确边界检测
- CLIP 零样本分类

### 2. 光谱指数计算
- NDVI（归一化植被指数）
- NDWI（归一化水体指数）
- NDBI（归一化建筑指数）
- SAVI（土壤调节植被指数）

### 3. 时序分析
- 变化检测
- 趋势分析
- 异常检测

### 4. 可视化
- 分割结果可视化
- 统计图表
- 交互式Web地图

## 📊 示例结果

输入遥感图像 → SAM分割 → CLIP分类 → 语义分割结果

## ⚙️ 高级配置

创建自定义配置文件 `config.yaml`:

```yaml
sam:
  model_type: "vit_h"
  device: "cuda"
  
processing:
  tile_size: 1024
  overlap: 128
  
classes:
  urban: ["building", "road", "parking"]
  vegetation: ["forest", "grassland", "cropland"]
  water: ["river", "lake", "ocean"]
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📜 许可证

MIT License

## 🙏 致谢

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything)
- [CLIP](https://github.com/openai/CLIP)
- [Rasterio](https://github.com/rasterio/rasterio)