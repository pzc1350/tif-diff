"""
快速开始脚本 - 一键运行遥感解译
"""

import os
from pathlib import Path

def download_sam_checkpoint():
    """下载SAM模型权重"""
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    checkpoint_path = "sam_vit_h_4b8939.pth"
    
    if not os.path.exists(checkpoint_path):
        print("下载SAM模型权重...")
        import urllib.request
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print("下载完成!")
    
    return checkpoint_path

def quick_process(image_path: str):
    """快速处理单张遥感图像"""
    from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM
    from visualization import RemoteSensingVisualizer
    
    # 下载模型
    checkpoint = download_sam_checkpoint()
    
    # 配置
    config = RemoteSensingConfig(
        sam_checkpoint=checkpoint,
        sam_model_type="vit_h"
    )
    
    # 初始化
    print("初始化模型...")
    rs_sam = RemoteSensingSAM(config)
    visualizer = RemoteSensingVisualizer()
    
    # 处理
    print("处理图像...")
    results = rs_sam.process_remote_sensing_image(image_path)
    
    # 可视化
    print("生成可视化...")
    output_dir = Path("quick_output")
    output_dir.mkdir(exist_ok=True)
    
    visualizer.visualize_results(
        results,
        save_path=output_dir / "result.png"
    )
    
    visualizer.export_results(
        results,
        output_dir=str(output_dir),
        formats=['geotiff', 'json']
    )
    
    print(f"\n✅ 处理完成！结果保存在: {output_dir}")
    
    return results

if __name__ == "__main__":
    # 使用示例
    # quick_process("path/to/your/remote_sensing_image.tif")
    
    # 或者处理示例数据
    import sys
    if len(sys.argv) > 1:
        quick_process(sys.argv[1])
    else:
        print("用法: python quick_start.py <图像路径>")