import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2

class FewShotRemoteSensing:
    """少样本遥感图像分割"""
    
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        
    def extract_features(self, image):
        """提取图像特征"""
        h, w, c = image.shape
        
        # 基础特征：颜色
        features = [image.reshape(-1, c)]
        
        # 纹理特征（使用Gabor滤波器）
        gabor_features = []
        for theta in np.arange(0, np.pi, np.pi/4):
            kernel = cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0)
            for i in range(c):
                filtered = cv2.filter2D(image[:,:,i], cv2.CV_32F, kernel)
                gabor_features.append(filtered.reshape(-1, 1))
        features.extend(gabor_features)
        
        # NDVI（如果有近红外波段）
        if c >= 4:  # 假设第4个波段是NIR
            nir = image[:,:,3].astype(float)
            red = image[:,:,0].astype(float)
            ndvi = (nir - red) / (nir + red + 1e-8)
            features.append(ndvi.reshape(-1, 1))
        
        # 合并所有特征
        all_features = np.hstack(features)
        
        return all_features
    
    def unsupervised_segmentation(self, image, reduce_dim=True):
        """无监督分割"""
        # 提取特征
        features = self.extract_features(image)
        
        # 降维（可选）
        if reduce_dim and features.shape[1] > 10:
            pca = PCA(n_components=10)
            features = pca.fit_transform(features)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # 重塑为图像
        h, w = image.shape[:2]
        segmentation = labels.reshape(h, w)
        
        return segmentation
    
    def interactive_refinement(self, image, initial_segmentation):
        """交互式细化（模拟少样本学习）"""
        # 这里可以添加用户交互逻辑
        # 例如：让用户标记几个样本点，然后传播标签
        
        # 简化示例：使用图割算法细化
        from skimage.segmentation import slic, mark_boundaries
        
        # 超像素分割
        segments = slic(image[:,:,:3], n_segments=500, compactness=10)
        
        # 基于初始分割结果，为每个超像素分配标签
        refined_segmentation = np.zeros_like(initial_segmentation)
        for segment_id in np.unique(segments):
            mask = segments == segment_id
            # 使用投票机制
            labels, counts = np.unique(initial_segmentation[mask], return_counts=True)
            refined_segmentation[mask] = labels[np.argmax(counts)]
        
        return refined_segmentation

# 使用示例
def quick_segment(image_path):
    """快速分割遥感图像"""
    import rasterio
    
    # 读取图像
    with rasterio.open(image_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    
    # 创建分割器
    segmenter = FewShotRemoteSensing(n_clusters=6)
    
    # 执行分割
    print("正在执行无监督分割...")
    segmentation = segmenter.unsupervised_segmentation(image)
    
    # 可选：交互式细化
    print("正在细化结果...")
    refined_segmentation = segmenter.interactive_refinement(image, segmentation)
    
    return refined_segmentation