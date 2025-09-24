import argparse
from pathlib import Path
import time
from datetime import datetime
import yaml

from sam_rs_core import RemoteSensingConfig, RemoteSensingSAM
from temporal_analysis import TemporalRemoteSensing
from visualization import RemoteSensingVisualizer

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description='高级遥感图像解译系统')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--mode', choices=['single', 'temporal', 'batch'], 
                       default='single', help='处理模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--visualize', action='store_true', help='是否可视化结果')
    parser.add_argument('--export-formats', nargs='+', 
                       default=['geotiff', 'json'],
                       choices=['geotiff', 'shp', 'json', 'html'],
                       help='导出格式')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = RemoteSensingConfig(**config_dict)
    else:
        config = RemoteSensingConfig()
        
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化模型
    print("初始化遥感解译系统...")
    rs_sam = RemoteSensingSAM(config)
    visualizer = RemoteSensingVisualizer()
    
    start_time = time.time()
    
    if args.mode == 'single':
        # 单张图像处理
        print(f"处理图像: {args.input}")
        results = rs_sam.process_remote_sensing_image(args.input)
        
    elif args.mode == 'temporal':
        # 时序分析
        temporal_analyzer = TemporalRemoteSensing(rs_sam)
        # 这里需要提供多个图像路径和日期
        # results = temporal_analyzer.analyze_temporal_series(image_paths, dates)
        
    elif args.mode == 'batch':
        # 批量处理
        input_dir = Path(args.input)
        image_files = list(input_dir.glob('*.tif'))
        print(f"找到 {len(image_files)} 个图像文件")
        
        results = []
        for img_path in image_files:
            result = rs_sam.process_remote_sensing_image(str(img_path))
            results.append(result)
            
    processing_time = time.time() - start_time
    print(f"处理完成，用时: {processing_time:.2f} 秒")
    
    # 添加处理时间到结果
    if isinstance(results, dict):
        results['processing_time'] = processing_time
    
    # 可视化
    if args.visualize:
        print("生成可视化...")
        visualizer.visualize_results(
            results,
            save_path=output_dir / 'visualization.png'
        )
        
        if 'html' in args.export_formats:
            visualizer.create_interactive_map(
                results,
                output_path=str(output_dir / 'interactive_map.html')
            )
    
    # 导出结果
    print("导出结果...")
    visualizer.export_results(
        results,
        output_dir=str(output_dir),
        formats=args.export_formats
    )
    
    print(f"\n所有结果已保存至: {output_dir}")
    
if __name__ == "__main__":
    main()