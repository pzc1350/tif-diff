#!/bin/bash

# 设置颜色
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== TIF-Diff 项目初始化 ===${NC}"
echo -e "${GREEN}开始创建项目文件...${NC}"

# 创建目录结构
echo -e "${YELLOW}创建目录结构...${NC}"
mkdir -p docs scripts examples tests configs

# 创建主要的 Python 文件
echo -e "${YELLOW}创建核心代码文件...${NC}"

# 1. 创建 README.md
cat > README.md << 'EOF'
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