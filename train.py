import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from src.data.dataloader import get_dataloader
from src.models.unet import UNet
from src.utils.metrics import calculate_metrics

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device).long()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, num_classes):
    """验证模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metrics = calculate_metrics(all_preds, all_labels, num_classes)
    
    return total_loss / len(dataloader), metrics

def main(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = get_dataloader(config)
    
    # 创建模型
    model = UNet(
        n_channels=config['model']['in_channels'],
        n_classes=config['data']['num_classes']
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['epochs']
    )
    
    # TensorBoard
    writer = SummaryWriter(config['paths']['log_dir'])
    
    # 创建检查点目录
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    best_iou = 0
    patience_counter = 0
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, metrics = validate(
            model, val_loader, criterion, device, config['data']['num_classes']
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录日志
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/mIoU', metrics['mean_iou'], epoch)
        writer.add_scalar('Metrics/accuracy', metrics['accuracy'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"mIoU: {metrics['mean_iou']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if metrics['mean_iou'] > best_iou:
            best_iou = metrics['mean_iou']
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': config
            }, checkpoint_dir / 'best_model.pth')
            
            print(f"Saved best model with mIoU: {best_iou:.4f}")
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练遥感图像分割模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    
    main(args.config)