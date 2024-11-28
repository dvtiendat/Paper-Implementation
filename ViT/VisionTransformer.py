import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patch = nn.Conv2d(in_channels, embedding_dim, patch_size, patch_size, 0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
    
    def forward(self, x):
        img_size = x.shape[-1]
        assert img_size % self.patch_size == 0, 'Image size must be divisible by patch size'
        x = self.patch(x)
        x = self.flatten(x)
        return x.permute(0, 2, 1)

class MSA(nn.Module):
    def __init__(self, 
                 embedding_dim=768, 
                 num_heads=12, 
                 attn_dropout=0):
        
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=768)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                                    num_heads=num_heads, 
                                                    dropout=attn_dropout, 
                                                    batch_first=True)
    
    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        
        return x

class MlpBlock(nn.Module):
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=768)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_size, embedding_dim),
            nn.Dropout(p=dropout)
        )
    
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, head_nums=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn_block = MSA(embedding_dim, head_nums, attn_dropout)
        self.mlp_block = MlpBlock(embedding_dim, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.multihead_attn_block(x) + x
        x = self.mlp_block(x) + x
        return x
    
class ViT(nn.Module):
    def __init__(self, 
                 img_size=224, 
                 in_channels=3, 
                 patch_size=16, 
                 num_transformer_layers=12, 
                 embed_dim=768,
                 mlp_size=3072,
                 num_heads=12,
                 attn_dropout=0,
                 mlp_dropout=0.1,
                 embedding_dropout=0.1,
                 n_classes=1000 
                 ):
        super().__init__()
        assert img_size % patch_size == 0, 'Image size must be divisible'
        self.num_patches = (img_size // patch_size)**2

        self.class_embedding = nn.Parameter(torch.rand(1, 1, embed_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, embed_dim), requires_grad=True)
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, mlp_size, mlp_dropout, attn_dropout) for _ in range(num_transformer_layers)]
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.cls(x[:, 0])

        return x
    
demo_img = torch.randn(1, 3, 224, 224)
print(demo_img.shape)