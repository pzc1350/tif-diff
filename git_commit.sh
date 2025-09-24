#!/bin/bash

echo "准备提交代码到 GitHub..."

# 确保在正确的目录
cd /path/to/tif-diff

# 初始化git（如果需要）
if [ ! -d ".git" ]; then
    git init
    git remote add origin https://github.com/pzc1350/tif-diff.git
fi

# 创建必要的目录
mkdir -p docs scripts examples tests configs

# 添加所有文件
git add .

# 提交
git commit -m "feat: 添加详细文档和代码注释

- 添加功能说明文档 (FEATURES.md)
- 添加快速使用指南 (QUICKSTART.md)  
- 添加API参考文档 (API.md)
- 为所有核心代码添加中文注释
- 改进代码结构和可读性
- 添加使用示例和最佳实践

文档包括：
- 完整的功能介绍
- 技术架构说明
- API使用参考
- 性能优化建议
- 常见问题解答"

# 推送到主分支
git push -u origin main

echo "✅ 提交完成！"