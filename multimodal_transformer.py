import torch
import torch.nn as nn
from einops import rearrange
import timm

class RemoteSensingTransformer(nn.Module):
    """多模态遥感Transformer"""
    
    def __init__(
        self,
        num_spectral_bands=10,  # 多光谱波段数
        num_classes=10,
        d_model=768,
        nhead=12,
        num_layers=12,
        patch_size=16
    ):
        super().__init__()
        
        # 光谱编码器
        self.spectral_encoder = nn.Linear(num_spectral_bands, d_model)
        
        # 空间编码器 (使用预训练的ViT)
        self.spatial_encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0  # 移除分类头
        )
        
        # 时序编码器 (如果有多时相数据)
        self.temporal_encoder = nn.LSTM(
            d_model, d_model // 2, 
            num_layers=2, 
            bidirectional=True,
            batch_first=True
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        
        # 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward=2048
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 分割头
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(d_model, d_model//2, 2, stride=2),
            nn.BatchNorm2d(d_model//2),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model//2, d_model//4, 2, stride=2),
            nn.BatchNorm2d(d_model//4),
            nn.ReLU(),
            nn.ConvTranspose2d(d_model//4, num_classes, 2, stride=2)
        )
        
    def forward(self, multispectral, rgb=None, temporal_seq=None):
        """
        Args:
            multispectral: (B, C, H, W) 多光谱图像
            rgb: (B, 3, H, W) RGB图像 (可选)
            temporal_seq: (B, T, C, H, W) 时间序列 (可选)
        """
        B, C, H, W = multispectral.shape
        
        # 1. 光谱特征提取
        spectral_features = rearrange(multispectral, 'b c h w -> b (h w) c')
        spectral_features = self.spectral_encoder(spectral_features)
        
        # 2. 空间特征提取 (如果有RGB)
        if rgb is not None:
            spatial_features = self.spatial_encoder(rgb)
            spatial_features = spatial_features.unsqueeze(1).expand(-1, H*W//256, -1)
            
            # 跨模态融合
            fused_features, _ = self.cross_attention(
                spectral_features, spatial_features, spatial_features
            )
        else:
            fused_features = spectral_features
            
        # 3. 时序特征提取 (如果有时间序列)
        if temporal_seq is not None:
            T = temporal_seq.shape[1]
            temporal_features = []
            for t in range(T):
                t_feat = rearrange(temporal_seq[:, t], 'b c h w -> b (h w) c')
                t_feat = self.spectral_encoder(t_feat)
                temporal_features.append(t_feat)
            temporal_features = torch.stack(temporal_features, dim=1)
            
            # LSTM处理时序
            lstm_out, _ = self.temporal_encoder(temporal_features.mean(dim=2))
            temporal_context = lstm_out.unsqueeze(1).expand(-1, H*W//256, -1)
            
            # 融合时序信息
            fused_features = fused_features + temporal_context
            
        # 4. 解码
        decoded = self.decoder(fused_features, fused_features)
        
        # 5. 重塑并生成分割图
        decoded = rearrange(decoded, 'b (h w) c -> b c h w', h=H//16, w=W//16)
        segmentation = self.segmentation_head(decoded)
        
        return segmentation

# 使用预训练权重
def load_pretrained_rs_transformer():
    """加载预训练的遥感Transformer"""
    model = RemoteSensingTransformer()
    
    # 加载在大规模遥感数据集上的预训练权重
    # checkpoint = torch.load('rs_transformer_pretrained.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    return model