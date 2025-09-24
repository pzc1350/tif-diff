import numpy as np
from typing import List, Tuple, Dict
import rasterio
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

class TemporalRemoteSensing:
    """时序遥感分析"""
    
    def __init__(self, sam_model: RemoteSensingSAM):
        self.sam_model = sam_model
        self.change_threshold = 0.3
        
    def analyze_temporal_series(
        self, 
        image_paths: List[str], 
        dates: List[datetime]
    ) -> Dict:
        """分析时间序列遥感图像"""
        print(f"分析 {len(image_paths)} 个时相的数据...")
        
        # 处理每个时相
        temporal_results = []
        for i, (path, date) in enumerate(zip(image_paths, dates)):
            print(f"处理时相 {i+1}/{len(image_paths)}: {date}")
            result = self.sam_model.process_remote_sensing_image(path)
            result['date'] = date
            temporal_results.append(result)
            
        # 变化检测
        changes = self._detect_changes(temporal_results)
        
        # 趋势分析
        trends = self._analyze_trends(temporal_results)
        
        # 异常检测
        anomalies = self._detect_anomalies(temporal_results)
        
        return {
            'temporal_results': temporal_results,
            'changes': changes,
            'trends': trends,
            'anomalies': anomalies
        }
    
    def _detect_changes(self, temporal_results: List[Dict]) -> List[Dict]:
        """检测变化"""
        changes = []
        
        for i in range(1, len(temporal_results)):
            prev_seg = temporal_results[i-1]['segmentation_map']
            curr_seg = temporal_results[i]['segmentation_map']
            
            # 计算变化图
            change_map = prev_seg != curr_seg
            
            # 分析变化类型
            change_analysis = self._analyze_change_types(prev_seg, curr_seg, change_map)
            
            changes.append({
                'from_date': temporal_results[i-1]['date'],
                'to_date': temporal_results[i]['date'],
                'change_map': change_map,
                'change_percentage': np.sum(change_map) / change_map.size,
                'change_analysis': change_analysis
            })
            
        return changes
    
    def _analyze_change_types(
        self, 
        prev_seg: np.ndarray, 
        curr_seg: np.ndarray,
        change_map: np.ndarray
    ) -> Dict:
        """分析变化类型"""
        # 类别名称
        categories = {
            0: 'unknown',
            1: 'urban',
            2: 'vegetation',
            3: 'water',
            4: 'bare_land'
        }
        
        change_matrix = np.zeros((5, 5))
        
        # 计算转移矩阵
        for i in range(5):
            for j in range(5):
                mask = (prev_seg == i) & (curr_seg == j) & change_map
                change_matrix[i, j] = np.sum(mask)
                
        # 分析主要变化
        major_changes = []
        for i in range(5):
            for j in range(5):
                if i != j and change_matrix[i, j] > 1000:  # 阈值可调
                    major_changes.append({
                        'from': categories[i],
                        'to': categories[j],
                        'pixels': int(change_matrix[i, j]),
                        'area_km2': change_matrix[i, j] * 0.0009  # 假设30m分辨率
                    })
                    
        return {
            'transition_matrix': change_matrix,
            'major_changes': sorted(major_changes, key=lambda x: x['pixels'], reverse=True)
        }
    
    def _analyze_trends(self, temporal_results: List[Dict]) -> Dict:
        """分析趋势"""
        # 提取每个类别的面积时间序列
        category_areas = {i: [] for i in range(5)}
        dates = []
        
        for result in temporal_results:
            seg_map = result['segmentation_map']
            dates.append(result['date'])
            
            for cat in range(5):
                area = np.sum(seg_map == cat)
                category_areas[cat].append(area)
                
        # 计算趋势
        trends = {}
        categories = ['unknown', 'urban', 'vegetation', 'water', 'bare_land']
        
        for cat, name in enumerate(categories):
            if cat == 0:  # 跳过unknown
                continue
                
            areas = np.array(category_areas[cat])
            if len(areas) > 2:
                # 线性回归
                x = np.arange(len(areas))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, areas)
                
                trends[name] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'trend': 'increasing' if slope > 0 else 'decreasing',
                    'significant': p_value < 0.05
                }
                
        return trends
    
    def _detect_anomalies(self, temporal_results: List[Dict]) -> List[Dict]:
        """检测异常"""
        anomalies = []
        
        # 基于光谱指数的异常检测
        for i, result in enumerate(temporal_results):
            indices = result['spectral_indices']
            
            # 检查NDVI异常
            if 'ndvi' in indices:
                ndvi = indices['ndvi']
                ndvi_mean = np.mean(ndvi[~np.isnan(ndvi)])
                ndvi_std = np.std(ndvi[~np.isnan(ndvi)])
                
                # Z-score异常检测
                if abs(ndvi_mean) > 3 * ndvi_std:
                    anomalies.append({
                        'date': result['date'],
                        'type': 'spectral_anomaly',
                        'index': 'NDVI',
                        'severity': 'high',
                        'details': f'NDVI mean: {ndvi_mean:.3f}'
                    })
                    
        return anomalies

class MultiModalFusion:
    """多模态数据融合"""
    
    def __init__(self):
        self.fusion_weights = None
        
    def fuse_optical_and_sar(
        self, 
        optical_data: np.ndarray, 
        sar_data: np.ndarray
    ) -> np.ndarray:
        """融合光学和SAR数据"""
        # PCA融合
        h, w = optical_data.shape[:2]
        
        # 展平数据
        optical_flat = optical_data.reshape(-1, optical_data.shape[-1])
        sar_flat = sar_data.reshape(-1, sar_data.shape[-1])
        
        # 合并数据
        combined = np.concatenate([optical_flat, sar_flat], axis=1)
        
        # PCA降维
        pca = PCA(n_components=min(10, combined.shape[1]))
        fused_flat = pca.fit_transform(combined)
        
        # 重塑
        fused = fused_flat.reshape(h, w, -1)
        
        return fused
    
    def fuse_multitemporal(
        self, 
        images: List[np.ndarray], 
        method: str = 'median'
    ) -> np.ndarray:
        """融合多时相图像"""
        stack = np.stack(images, axis=0)
        
        if method == 'median':
            return np.median(stack, axis=0)
        elif method == 'mean':
            return np.mean(stack, axis=0)
        elif method == 'max':
            return np.max(stack, axis=0)
        else:
            raise ValueError(f"未知的融合方法: {method}")