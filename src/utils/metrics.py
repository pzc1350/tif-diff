import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(predictions, labels, num_classes):
    """计算语义分割的评估指标"""
    
    # 展平数组
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions, labels=list(range(num_classes)))
    
    # 计算IoU
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou = intersection / (union + 1e-10)
    
    # 计算mIoU
    mean_iou = np.mean(iou[union > 0])
    
    # 计算准确率
    accuracy = np.sum(intersection) / np.sum(cm)
    
    # 计算精确率和召回率
    precision = intersection / (cm.sum(axis=0) + 1e-10)
    recall = intersection / (cm.sum(axis=1) + 1e-10)
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    return {
        'iou_per_class': iou,
        'mean_iou': mean_iou,
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'confusion_matrix': cm
    }

def print_metrics(metrics, class_names):
    """打印评估指标"""
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    
    print("\nPer-class Metrics:")
    print(f"{'Class':<15} {'IoU':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)
    
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15} "
              f"{metrics['iou_per_class'][i]:<10.4f} "
              f"{metrics['precision_per_class'][i]:<10.4f} "
              f"{metrics['recall_per_class'][i]:<10.4f} "
              f"{metrics['f1_per_class'][i]:<10.4f}")