from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ResamplerProjectorConfig:
    type: str = "resampler"
    num_heads: int = 8
    mlp_ratio: float = 2.0
    num_query_tokens: int = 8
    depth: int = 4
    gradient_checkpointing: bool = False
    qk_norm: bool = True
    is_gated: bool = False


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_heads: int,
        qk_norm: bool = True,
        is_gated: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.qk_norm = qk_norm
        self.is_gated = is_gated

        self.norm1 = nn.RMSNorm(in_features)  # image features
        self.norm2 = nn.RMSNorm(in_features)  # latent queries

        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim)
            self.norm_k = nn.RMSNorm(self.head_dim)

        self.to_q = nn.Linear(in_features, in_features, bias=False)
        self.to_k = nn.Linear(in_features, in_features, bias=False)
        self.to_v = nn.Linear(in_features, in_features, bias=False)
        self.to_out = nn.Linear(in_features, in_features, bias=False)

        if is_gated:
            self.to_gate = nn.Linear(in_features, in_features, bias=False)

    def _pre_attn_reshape(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    # def _post_attn_reshape(self, tensor: torch.Tensor) -> torch.Tensor:
    #     batch_size, _num_heads, seq_len, _head_dim = tensor.shape
    #     return tensor.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.in_features)

    def forward(self, image_features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        image_features = self.norm1(image_features)
        latents = self.norm2(latents)

        kv_input = torch.cat([image_features, latents], dim=1)
        query = self.to_q(latents)
        key = self.to_k(kv_input)
        value = self.to_v(kv_input)

        query = self._pre_attn_reshape(query)
        key = self._pre_attn_reshape(key)
        value = self._pre_attn_reshape(value)

        if self.qk_norm:
            query = self.norm_q(query)
            key = self.norm_k(key)

        attn = F.scaled_dot_product_attention(query, key, value, is_causal=False)

        batch_size, num_query_tokens, _ = latents.shape
        attn = attn.permute(0, 2, 1, 3).contiguous()
        if self.is_gated:
            gate = self.to_gate(latents).reshape(
                batch_size,
                num_query_tokens,
                self.num_heads,
                self.head_dim,
            )
            attn = attn * torch.sigmoid(gate)
        attn = attn.view(batch_size, num_query_tokens, self.in_features)

        return self.to_out(attn)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ResamplerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qk_norm: bool = True,
        is_gated: bool = False,
    ):
        super().__init__()
        self.attn = PerceiverAttention(
            in_features=dim,
            num_heads=num_heads,
            qk_norm=qk_norm,
            is_gated=is_gated,
        )
        self.mlp = SwiGLU(dim=dim, hidden_dim=int(dim * mlp_ratio))

    def forward(self, image_features: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        latents = latents + self.attn(image_features, latents)
        latents = latents + self.mlp(latents)
        return latents


# Adapted from https://github.com/tencent-ailab/IP-Adapter/blob/62e4af9d0c1ac7d5f8dd386a0ccf2211346af1a2/ip_adapter/resampler.py#L81
class ResamplerProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        num_query_tokens: int = 8,
        depth: int = 4,
        gradient_checkpointing: bool = False,
        qk_norm: bool = True,
        is_gated: bool = False,
    ):
        super().__init__()

        self.num_query_tokens = num_query_tokens
        self.out_dim = out_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.queries = nn.Parameter(torch.randn(1, num_query_tokens, out_dim) / out_dim**0.5)
        self.proj_in = nn.Linear(in_dim, out_dim)
        self.proj_out = nn.Linear(out_dim, out_dim)
        self.norm_out = nn.RMSNorm(out_dim)

        self.blocks = nn.ModuleList(
            [
                ResamplerBlock(
                    dim=out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    is_gated=is_gated,
                )
                for _ in range(depth)
            ]
        )

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.blocks.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.RMSNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)

        self.queries.data = torch.randn(1, self.num_query_tokens, self.out_dim) / self.out_dim**0.5

        for linear in (self.proj_in, self.proj_out):
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        if self.norm_out.weight is not None:
            nn.init.ones_(self.norm_out.weight)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size = image_features.size(0)
        latents = self.queries.expand(batch_size, -1, -1)
        image_features = self.proj_in(image_features)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                latents = checkpoint(block, image_features, latents, use_reentrant=False)
            else:
                latents = block(image_features, latents)

        return self.norm_out(self.proj_out(latents))
