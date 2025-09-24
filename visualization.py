import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from rasterio.plot import show
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import box, Polygon
import json

class RemoteSensingVisualizer:
    """遥感结果可视化"""
    
    def __init__(self):
        # 定义配色方案
        self.colors = {
            'unknown': '#808080',
            'urban': '#FF0000',
            'vegetation': '#00FF00',
            'water': '#0000FF',
            'bare_land': '#A52A2A'
        }
        
        self.cmap = ListedColormap([
            self.colors['unknown'],
            self.colors['urban'],
            self.colors['vegetation'],
            self.colors['water'],
            self.colors['bare_land']
        ])
        
    def visualize_results(
        self, 
        results: Dict, 
        save_path: str = None,
        show_indices: bool = True
    ):
        """综合可视化结果"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 原始RGB图像
        ax1 = plt.subplot(2, 3, 1)
        # 这里需要从results中获取RGB图像
        ax1.set_title('原始图像', fontsize=14)
        ax1.axis('off')
        
        # 2. 语义分割结果
        ax2 = plt.subplot(2, 3, 2)
        seg_map = results['segmentation_map']
        im = ax2.imshow(seg_map, cmap=self.cmap, interpolation='nearest')
        ax2.set_title('语义分割', fontsize=14)
        ax2.axis('off')
        
        # 添加图例
        legend_elements = [
            mpatches.Patch(color=color, label=label.title())
            for label, color in self.colors.items()
        ]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 3. NDVI
        if show_indices and 'spectral_indices' in results:
            if 'ndvi' in results['spectral_indices']:
                ax3 = plt.subplot(2, 3, 3)
                ndvi = results['spectral_indices']['ndvi']
                im3 = ax3.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
                ax3.set_title('NDVI', fontsize=14)
                ax3.axis('off')
                plt.colorbar(im3, ax=ax3, fraction=0.046)
                
        # 4. 掩码数量统计
        ax4 = plt.subplot(2, 3, 4)
        if 'masks' in results:
            categories = {}
            for mask in results['masks']:
                cat = mask.get('category', 'unknown')
                categories[cat] = categories.get(cat, 0) + 1
                
            bars = ax4.bar(categories.keys(), categories.values())
            for bar, (cat, _) in zip(bars, categories.items()):
                bar.set_color(self.colors.get(cat, '#808080'))
            ax4.set_title('类别分布', fontsize=14)
            ax4.set_xlabel('类别')
            ax4.set_ylabel('掩码数量')
            
        # 5. 面积统计
        ax5 = plt.subplot(2, 3, 5)
        area_stats = self._calculate_area_statistics(seg_map)
        
        labels = list(area_stats.keys())
        sizes = list(area_stats.values())
        colors = [self.colors.get(label, '#808080') for label in labels]
        
        ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        ax5.set_title('面积占比', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def _calculate_area_statistics(self, seg_map: np.ndarray) -> Dict[str, float]:
        """计算各类别面积统计"""
        categories = ['unknown', 'urban', 'vegetation', 'water', 'bare_land']
        stats = {}
        
        total_pixels = seg_map.size
        for i, cat in enumerate(categories):
            count = np.sum(seg_map == i)
            if count > 0:
                stats[cat] = count / total_pixels * 100
                
        return stats
    
    def create_interactive_map(
        self, 
        results: Dict, 
        output_path: str = 'map.html'
    ):
        """创建交互式地图"""
        # 获取地理范围
        metadata = results.get('metadata', {})
        bounds = metadata.get('bounds', None)
        
        if bounds is None:
            print("警告：无法获取地理坐标信息")
            return
            
        # 创建地图
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=13,
            control_scale=True
        )
        
        # 添加分割结果作为图层
        # 这里需要将分割结果转换为GeoJSON
        geojson_data = self._segmentation_to_geojson(
            results['segmentation_map'],
            results['metadata']
        )
        
        # 添加GeoJSON图层
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': self.colors.get(feature['properties']['class'], '#808080'),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7,
            }
        ).add_to(m)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        # 保存地图
        m.save(output_path)
        print(f"交互式地图已保存至: {output_path}")
        
    def _segmentation_to_geojson(
        self, 
        seg_map: np.ndarray, 
        metadata: Dict
    ) -> Dict:
        """将分割结果转换为GeoJSON"""
        # 简化实现，实际应该使用rasterio的特征提取
        features = []
        
        # 这里需要实现将栅格转换为矢量的逻辑
        # 使用rasterio.features.shapes等
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def export_results(
        self, 
        results: Dict, 
        output_dir: str,
        formats: List[str] = ['geotiff', 'shp', 'json']
    ):
        """导出结果到多种格式"""
        from pathlib import Path
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 1. 导出GeoTIFF
        if 'geotiff' in formats:
            self._export_geotiff(
                results['segmentation_map'],
                results['metadata'],
                output_dir / 'segmentation.tif'
            )
            
        # 2. 导出Shapefile
        if 'shp' in formats:
            self._export_shapefile(
                results['segmentation_map'],
                results['metadata'],
                output_dir / 'segmentation.shp'
            )
            
        # 3. 导出JSON统计
        if 'json' in formats:
            stats = {
                'area_statistics': self._calculate_area_statistics(results['segmentation_map']),
                'metadata': results['metadata'],
                'processing_time': results.get('processing_time', 'N/A')
            }
            
            with open(output_dir / 'statistics.json', 'w') as f:
                json.dump(stats, f, indent=2)
                
    def _export_geotiff(
        self, 
        seg_map: np.ndarray, 
        metadata: Dict,
        output_path: str
    ):
        """导出为GeoTIFF"""
        # 使用原始的地理参考信息
        transform = metadata.get('transform', None)
        crs = metadata.get('crs', None)
        
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=seg_map.shape[0],
            width=seg_map.shape[1],
            count=1,
            dtype=seg_map.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(seg_map, 1)
            
        print(f"GeoTIFF已导出: {output_path}")