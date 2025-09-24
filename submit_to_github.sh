#!/bin/bash

# 设置颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}开始提交代码到 GitHub...${NC}"

# 创建项目目录结构
echo -e "${GREEN}创建目录结构...${NC}"
mkdir -p docs
mkdir -p scripts
mkdir -p examples
mkdir -p tests
mkdir -p configs
mkdir -p weights

# 创建 .gitignore 文件
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# 权重文件
weights/
*.pth
*.ckpt
*.onnx

# 数据文件
data/
*.tif
*.tiff
*.geotiff
*.shp
*.shx
*.dbf
*.prj
*.cpg

# 输出文件
output/
results/
quick_output/
*.png
*.jpg
*.html
*.json
*.csv

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# 临时文件
*.tmp
*.bak
temp/
cache/
EOF

# 初始化 Git 仓库
if [ ! -d ".git" ]; then
    echo -e "${GREEN}初始化 Git 仓库...${NC}"
    git init
    git branch -M main
    git remote add origin https://github.com/pzc1350/tif-diff.git
fi

# 添加所有文件
echo -e "${GREEN}添加文件到 Git...${NC}"
git add .

# 创建提交
echo -e "${GREEN}创建提交...${NC}"
git commit -m "feat: 实现基于SAM的遥感图像智能解译系统

🚀 核心功能：
- 零样本语义分割（SAM + CLIP）
- 多光谱指数计算（NDVI/NDWI/NDBI/SAVI）
- 大图像分块处理
- 时序分析与变化检测
- 丰富的可视化功能

📁 项目结构：
- sam_rs_core.py: 核心SAM处理模块
- temporal_analysis.py: 时序分析
- visualization.py: 可视化工具
- quick_start.py: 快速开始脚本

📚 文档：
- 完整的中文注释
- API参考文档
- 功能说明文档
- 快速使用指南

🛠 技术栈：
- Segment Anything Model (SAM)
- CLIP零样本分类
- Rasterio遥感数据处理
- PyTorch深度学习框架"

# 推送到 GitHub
echo -e "${GREEN}推送到 GitHub...${NC}"
git push -u origin main --force

echo -e "${GREEN}✅ 代码已成功提交到 https://github.com/pzc1350/tif-diff${NC}"
echo -e "${BLUE}您可以访问仓库查看提交的代码${NC}"