#!/bin/bash

# 初始化 git (如果还没有)
git init

# 添加所有文件
git add .

# 提交
git commit -m "feat: 实现基于SAM的先进遥感图像解译系统

- 添加 SAM + CLIP 零样本语义分割
- 实现多光谱指数计算 (NDVI, NDWI, NDBI, SAVI)
- 支持大图像分块处理
- 添加时序分析功能
- 实现丰富的可视化和多格式导出
- 添加完整的文档和示例"

# 添加远程仓库 (如果还没有)
git remote add origin https://github.com/pzc1350/tif-diff.git

# 推送到主分支
git push -u origin main