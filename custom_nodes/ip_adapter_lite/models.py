"""
Minimal model classes for IP-Adapter Lite (SD1.5, low-VRAM).

Only the standard ImageProjModel is implemented (not Resampler / Plus variants),
which keeps the memory footprint small and avoids heavy dependency on einops.
"""

import torch
import torch.nn as nn


class ImageProjModel(nn.Module):
    """
    Linear projection from CLIP pooled image embeddings into IP-Adapter token space.

    Mirrors the original ImageProjModel from ip_adapter/ip_adapter.py:
        Linear(clip_dim -> num_tokens * cross_attn_dim) + LayerNorm
    """

    def __init__(
        self,
        cross_attention_dim: int = 768,
        clip_embeddings_dim: int = 1024,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.proj = nn.Linear(clip_embeddings_dim, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, clip_dim]  →  out: [batch, num_tokens, cross_attn_dim]
        x = self.proj(x)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        return self.norm(x)


class IPToKV(nn.Module):
    """
    Holds per-UNet-layer to_k_ip / to_v_ip projection Linear layers.

    State dict keys like "1.to_k_ip.weight" are stored under the name
    "1_to_k_ip" (dots replaced by underscores, ".weight" stripped) so they
    can live inside a ModuleDict.
    """

    def __init__(self, state_dict: dict):
        super().__init__()
        self.layers = nn.ModuleDict()
        for key, weight in state_dict.items():
            name = key.replace(".weight", "").replace(".", "_")
            linear = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
            linear.weight.data = weight.clone()
            self.layers[name] = linear
