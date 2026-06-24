import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    """Блок MLP-Mixer с residual connection."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class MLP_Mixer(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=512, num_layers=6, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        input_dim = latent_dim * 3  # diff + z1 + z2

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MixerBlock(hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden_dim, latent_dim)
        # residual_scale убран, так как предсказываем консенсус напрямую

    def forward(self, diff: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        diff: [B, N, latent_dim]
        z1, z2: [B, N, latent_dim]
        Returns: z_cons [B, N, latent_dim]
        """
        x = torch.cat([diff, z1, z2], dim=-1)
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
            
        z_cons = self.head(x)
        return z_cons