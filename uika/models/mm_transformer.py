import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
assert hasattr(F, "scaled_dot_product_attention")

from functools import partial
from typing import Any, Dict, Optional, Tuple, Literal
from diffusers.utils import is_torch_version
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import JointAttnProcessor2_0


def get_2d_sincos_pos_embed(embed_dim, grid_size, add_cls_token=False):
    """
    Create 2D sin/cos positional embeddings.

    Args:
        embed_dim (`int`):
            Embedding dimension.
        grid_size (`int`):
            The grid height and width.
        add_cls_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a classification (CLS) token.

    Returns:
        (`torch.FloatTensor` of shape (grid_size*grid_size, embed_dim) or (1+grid_size*grid_size, embed_dim): the
        position embeddings (with or without classification token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if add_cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")

    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MMTransformer(nn.Module):
    """
    Transformer blocks that process the input and optionally use condition and modulation.
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        inner_dim: int,
        cond_dim: int = None,
        cond_dim2: int = None,
        gradient_checkpointing=False,
        eps: float = 1e-6,
        use_dual_attention: bool = True,
        uv_token_size: Literal[64, 96, 128] = 64,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_dual_attention = use_dual_attention

        assert num_layers % 4 == 0
        self.skip_step = num_layers // 4
        
        # modules
        self.layers = nn.ModuleList([
            self._block_fn(inner_dim, cond_dim, cond_dim2)(
                eps=eps,
                num_attention_heads=num_heads,
                context_pre_only=(i == num_layers - 1),
                use_dual_attention=use_dual_attention,
            )
            for i in range(num_layers)
        ])
        self.norm_list = nn.ModuleList([nn.LayerNorm(inner_dim, eps=eps) for _ in range(4)])
        self.linear_cond_proj = nn.Linear(cond_dim, inner_dim)
        if use_dual_attention:
            self.linear_cond_proj2 = nn.Linear(cond_dim2, inner_dim)
        
        # learnable token & pos embed
        self.uv_token = nn.Parameter(torch.randn(uv_token_size * uv_token_size, inner_dim))
        nn.init.trunc_normal_(self.uv_token, std=0.02)
        self.uv_pos_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(inner_dim, uv_token_size)).float())

    def _block_fn(self, inner_dim, cond_dim, cond_dim2):
        assert inner_dim is not None, f"inner_dim must always be specified"
        assert inner_dim == cond_dim
        if self.use_dual_attention:
            assert inner_dim == cond_dim2
        return partial(MMTransformerBlock, dim=inner_dim, qk_norm="rms_norm")

    def _get_query(self, batch_size: int) -> torch.Tensor:
        B = batch_size
        x = self.uv_token.expand(B, -1, -1)
        pos_embed = self.uv_pos_embed.expand(B, -1, -1)
        return x + pos_embed

    def forward(
        self,
        cond: torch.Tensor,  # [B, L_cond, D_cond]
        cond2: Optional[torch.FloatTensor] = None  # [B, L_cond, D_cond]
    ):
        x = self._get_query(cond.shape[0])
        x = x.to(cond.dtype)

        cond = self.linear_cond_proj(cond)
        if self.use_dual_attention:
            if cond2 is None:
                raise ValueError("cond2 is required when use_dual_attention=True.")
            cond2 = self.linear_cond_proj2(cond2)
        elif cond2 is not None:
            raise ValueError("cond2 was provided, but use_dual_attention=False.")
        
        x_list = []
        intermediate_count = 0
        
        for layer_idx, layer in enumerate(self.layers):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                x, cond, cond2 = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    cond,
                    cond2,
                    **ckpt_kwargs,
                )
            else:
                x, cond, cond2 = layer(
                    hidden_states=x,
                    encoder_hidden_states=cond,
                    encoder_hidden_states2=cond2,
                )
            if (layer_idx + 1) % self.skip_step == 0:
                x_list.append(self.norm_list[intermediate_count](x))
                intermediate_count += 1
        
        assert len(x_list) == 4
        return x_list


class MMTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """
    def __init__(
        self,
        dim: int,
        eps: float,
        num_attention_heads: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        use_dual_attention: bool = True,
    ):
        super().__init__()
        attention_head_dim = dim // num_attention_heads
        assert attention_head_dim * num_attention_heads == dim
        
        self.use_dual_attention = use_dual_attention
        self.context_pre_only = context_pre_only

        self.norm1 = nn.LayerNorm(dim)
        self.norm1_context = nn.LayerNorm(dim)

        processor = JointAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        if use_dual_attention:
            self.norm1_context2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                processor=processor,
                qk_norm=qk_norm,
                eps=eps,
            )
        else:
            self.attn2 = None

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            if use_dual_attention:
                self.norm2_context2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
                self.ff_context2 = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None
            if use_dual_attention:
                self.norm2_context2 = None
                self.ff_context2 = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_hidden_states2: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor]:
        norm_hidden_states = self.norm1(hidden_states)
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        
        # Attention.
        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            norm_encoder_hidden_states2 = self.norm1_context2(encoder_hidden_states2)
            attn_output2, context_attn_output2 = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states2)
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
            if self.use_dual_attention:
                encoder_hidden_states2 = None
        else:
            encoder_hidden_states = encoder_hidden_states + context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + context_ff_output
            if self.use_dual_attention:
                encoder_hidden_states2 = encoder_hidden_states2 + context_attn_output2
                norm_encoder_hidden_states2 = self.norm2_context2(encoder_hidden_states2)
                context_ff_output2 = self.ff_context2(norm_encoder_hidden_states2)
                encoder_hidden_states2 = encoder_hidden_states2 + context_ff_output2

        return hidden_states, encoder_hidden_states, encoder_hidden_states2
