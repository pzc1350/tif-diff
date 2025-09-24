#!/usr/bin/env python
"""
下载 SAM 模型权重脚本
"""

import os
import urllib.request
from pathlib import Path
from tqdm import tqdm

# 模型下载链接
MODEL_URLS = {
    "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

def download_with_progress(url: str, filepath: str):
    """带进度条的下载"""
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                           desc=filepath.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filepath, reporthook=t.update_to)

def main():
    # 创建权重目录
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)
    
    print("SAM 模型权重下载工具")
    print("=" * 50)
    
    # 选择模型
    print("\n可用模型:")
    for i, (name, url) in enumerate(MODEL_URLS.items(), 1):
        size = {"sam_vit_h": "2.4GB", "sam_vit_l": "1.2GB", "sam_vit_b": "375MB"}
        print(f"{i}. {name} ({size.get(name, 'Unknown')})")
    
    choice = input("\n选择要下载的模型 (默认: 1): ") or "1"
    
    try:
        model_name = list(MODEL_URLS.keys())[int(choice) - 1]
    except (ValueError, IndexError):
        print("无效选择，使用默认模型")
        model_name = "sam_vit_h"
    
    # 下载
    url = MODEL_URLS[model_name]
    filepath = weights_dir / f"{model_name}.pth"
    
    if filepath.exists():
        print(f"\n{model_name} 已存在，跳过下载")
    else:
        print(f"\n开始下载 {model_name}...")
        download_with_progress(url, str(filepath))
        print(f"\n✅ 下载完成: {filepath}")
    
    # 创建默认符号链接
    default_link = weights_dir / "sam_vit_h_4b8939.pth"
    if not default_link.exists() and model_name == "sam_vit_h":
        os.symlink(filepath.name, str(default_link))
        print(f"创建默认链接: {default_link}")

if __name__ == "__main__":
    main()