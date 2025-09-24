import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

class ContrastiveRemoteSensingModel(nn.Module):
    """对比学习的遥感模型"""
    
    def __init__(self, encoder, projection_dim=256):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.num_features, encoder.num_features),
            nn.ReLU(),
            nn.Linear(encoder.num_features, projection_dim)
        )
        
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return F.normalize(projections, dim=-1)

class RemoteSensingMAE(nn.Module):
    """遥感图像的Masked Autoencoder"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=10, embed_dim=768):
        super().__init__()
        
        # 编码器
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, nhead=12, dim_feedforward=3072)
            for _ in range(12)
        ])
        
        # 解码器
        self.decoder_embed = nn.Linear(embed_dim, 512)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(512, nhead=8, dim_feedforward=2048)
            for _ in range(8)
        ])
        self.decoder_pred = nn.Linear(512, patch_size**2 * in_chans)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 512))
        
    def random_masking(self, x, mask_ratio=0.75):
        """随机遮盖patches"""
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留未遮盖的patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
        
    def forward(self, imgs, mask_ratio=0.75):
        # Patch embedding
        x = self.patch_embed(imgs)
        x = x.flatten(2).transpose(1, 2)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(imgs.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # 随机遮盖
        x, mask, ids_restore = self.random_masking(x[:, 1:], mask_ratio)
        x = torch.cat((x[:, :1], x), dim=1)
        
        # 编码
        for blk in self.blocks:
            x = blk(x)
            
        # 解码
        x = self.decoder_embed(x)
        
        # 添加mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat((x[:, 1:], mask_tokens), dim=1)
        x = torch.cat((x[:, :1], x_), dim=1)
        
        # 解码器
        for blk in self.decoder_blocks:
            x = blk(x)
            
        # 预测
        x = self.decoder_pred(x[:, 1:])
        
        return x, mask

# 训练函数
def pretrain_mae(model, dataloader, epochs=100):
    """预训练MAE模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    
    for epoch in range(epochs):
        for batch in dataloader:
            imgs = batch['image']
            
            # 前向传播
            pred, mask = model(imgs)
            
            # 计算损失 (只在被遮盖的部分)
            target = imgs.flatten(2).transpose(1, 2)
            loss = (pred - target) ** 2
            loss = (loss * mask).sum() / mask.sum()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()