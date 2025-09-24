import torch
import torch.nn as nn
from typing import Dict, List, Optional

class IntegratedRemoteSensingSystem(nn.Module):
    """集成的遥感解译系统"""
    
    def __init__(self):
        super().__init__()
        
        # 1. 特征提取backbone
        self.backbone = timm.create_model('convnext_base', pretrained=True, in_chans=10)
        
        # 2. 任务特定头
        self.heads = nn.ModuleDict({
            'segmentation': SegmentationHead(num_classes=10),
            'change_detection': ChangeDetectionHead(),
            'super_resolution': SuperResolutionHead(scale_factor=4),
            'cloud_removal': CloudRemovalHead(),
            'classification': ClassificationHead(num_classes=50)
        })
        
        # 3. 任务路由器 (动态选择任务)
        self.task_router = TaskRouter(num_tasks=5)
        
        # 4. 自适应融合模块
        self.fusion_module = AdaptiveFusion()
        
    def forward(
        self,
        x: torch.Tensor,
        tasks: List[str],
        metadata: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        多任务前向传播
        
        Args:
            x: 输入遥感图像
            tasks: 要执行的任务列表
            metadata: 元数据（坐标、时间等）
        """
        # 提取共享特征
        features = self.backbone(x)
        
        # 任务路由
        task_weights = self.task_router(features, tasks)
        
        # 执行各个任务
        outputs = {}
        for task in tasks:
            if task in self.heads:
                task_features = features * task_weights[task].unsqueeze(-1).unsqueeze(-1)
                outputs[task] = self.heads[task](task_features, metadata)
                
        # 自适应融合（如果需要）
        if len(tasks) > 1:
            outputs['fused'] = self.fusion_module(outputs, features)
            
        return outputs

class TaskRouter(nn.Module):
    """任务路由器：学习为不同任务分配权重"""
    
    def __init__(self, num_tasks):
        super().__init__()
        self.routing_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, num_tasks),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features, tasks):
        weights = self.routing_network(features)
        return {task: weights[:, i] for i, task in enumerate(tasks)}

# 使用示例
def run_integrated_system():
    """运行集成系统"""
    model = IntegratedRemoteSensingSystem()
    
    # 输入数据
    multispectral_image = torch.randn(1, 10, 512, 512)
    
    # 执行多个任务
    outputs = model(
        multispectral_image,
        tasks=['segmentation', 'change_detection', 'cloud_removal'],
        metadata={'timestamp': '2025-01-24', 'location': [120.0, 30.0]}
    )
    
    # 获取结果
    segmentation_map = outputs['segmentation']
    change_map = outputs['change_detection']
    clean_image = outputs['cloud_removal']