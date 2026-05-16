from __future__ import annotations
import json, math, os
import einx

import torch
import torch.nn as nn
from torch import Tensor

from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

from .nn_utils import softmax


# 注意输入顺序是in_features、out_features，线性层形状是(out_features, in_features)
class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 创建空tensor然后初始化
        weight = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = math.sqrt(2 / (in_features + out_features))
        # 加了_是原地修改，所以不用在前面写weigth=...
        torch.nn.init.trunc_normal_(weight, mean=0, std=std, a=-3 * std, b=3 * std)
        # 使得pytorch可以处理参数
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 等价于 Y = x @ W.T
        Y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return Y


# token -> 一个embedding_dim维张量的映射表
class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,  # 词汇表大小
        embedding_dim: int,  # 嵌入向量的维度，即d_model
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        weight = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        torch.nn.init.trunc_normal_(weight, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(weight)

    # 输入：(batch,seq)
    # 输出：(batch,seq,embedding_dim)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 最后自动按照token_ids里的内容，缩影embedding里的内容并进行合并
        # 即保留token_ids形状，在后面多加一维embedding内容
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.eps = eps

        g = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(g)

    def forward(self, x: torch.Tensor) -> Float[Tensor, " ... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x里每个元素都求平方
        x_square = x.pow(2)
        # 求平均值
        mean_square = x_square.mean(dim=-1, keepdim=True)
        # 加上eps再开根号，rms.shape == (batch, seq, 1)，后续x/rms时，
        rms = torch.sqrt(mean_square + self.eps)
        # 批量计算
        rms_norm = x / rms * self.weight
        # 恢复原本的类型
        return rms_norm.to(in_dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        d_k: int,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # d_k不是偶数直接报错
        assert d_k % 2 == 0

        self.d_k = d_k  # 每个token向量有多长
        self.max_seq_len = max_seq_len  # 序列有多长，即有多少个token

        # 计算i和j，即位置信息。j=k-1
        positions = torch.arange(max_seq_len, device=device, dtype=dtype)
        pair_indices = torch.arange(d_k // 2, device=device, dtype=dtype)
        positions = positions[:, None]
        pair_indices = pair_indices[None, :]

        # 构造cos表和sin表
        angles = positions / (theta ** (2 * pair_indices / d_k))
        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        # register_buffer存储
        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> Float[Tensor, " ... sequence_length d_k"]:
        # 获取保存的cos表和sin表
        cos = self.cos_table[token_positions]
        sin = self.sin_table[token_positions]

        # x.shape==(..., seq_len, d_k)
        # token_positions.shape == (..., seq_len)

        # 拆分x的最后一维为pair
        # x.shape==(..., seq_len, d_k//2, 2)
        x = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x1 = x[..., 0]
        x2 = x[..., 1]

        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        # rotated.shaped == (..., seq_len, d_k//2, 2)
        rotated = torch.stack((rot1, rot2), dim=-1)
        # rotated.shaped == (..., seq_len, d_k)
        rotated = rotated.reshape(*rotated.shape[:-2], self.d_k)
        return rotated


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x * torch.sigmoid(x)
        return result


# 注意这里(... d_model)经过w1、w3后变成(... d_ff)，最后通过w2变回(... d_model)
class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # w1,w3,w2用 Linear实现
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.silu(self.w1(x))
        w3_out = self.w3(x)
        w2_in = w1_out * w3_out
        Y = self.w2(w2_in)
        return Y


class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = None,
        max_seq_len: int = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model不能整除num_heads!!!")
        d_k = d_v = d_model // num_heads  # 整数除法
        # d_k必须为偶数
        if d_k % 2 != 0:
            raise ValueError("d_k不是偶数!!!")

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads

        # in_features = out_features = d_model
        self.o_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # 前d_model行是 W_q，中间是W_k，最后是W_v
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)

        # 看需求是否创建rope
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(
                d_k, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        is_rope: bool = False,
        token_positions: torch.Tensor = None,
    ) -> Float[Tensor, " ... sequence_length d_model"]:
        # 做整体weight和x的线性变换
        x_out = self.qkv_proj(x)
        qkv = rearrange(  # attention 期望的形状是(... seq_len, d_k)
            x_out,
            "... seq_len (three num_heads d_k) -> ... three num_heads seq_len d_k",
            num_heads=self.num_heads,
            d_k=self.d_k,
            three=3,
        )
        # 后面三维是 num_heads seq_len d_k
        q = qkv[..., 0, :, :, :]
        k = qkv[..., 1, :, :, :]
        v = qkv[..., 2, :, :, :]
        # 如果存在token_positions，就对q和k进行rope
        if is_rope == True:
            assert (
                self.rope is not None
            ), "在token_positions存在的情况下，self.rope不存在!!!"

            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 构造因果掩码 causal mask，形状为 (seq_len,seq_len)
        seq_len = x.shape[-2]
        mask = torch.tril(  # tril()只保留下三角和对角线
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        # 计算注意力分数
        attn_score = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_score = rearrange(
            attn_score, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)"
        )
        output = self.o_proj(attn_score)
        return output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = None,
        device: torch.device | None = None,
        dtype: torch.detype | None = None,
    ):
        super().__init__()

        self.rms_norm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiheadSelfAttention(
            d_model,
            num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.rms_norm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor
    ) -> Float[Tensor, " batch sequence_length d_model"]:

        # 前半部分
        normed_x = self.rms_norm_1(x)
        attn_out = self.attn(normed_x, is_rope=True)
        y = attn_out + x

        # 后半部分
        normed_y = self.rms_norm_2(y)
        ffn_out = self.ffn(normed_y)
        z = y + ffn_out

        return z


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,  # RoPE 所用的max_seq_len
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,  # RoPE 所用的theta
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        # 嵌入层
        self.embedding = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # 多个transformer块
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
            )
        # RMSNorm层
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        # 最后的线性层
        self.lm_head = Linear(  # 注意in_features和out_features
            d_model, vocab_size, device=device, dtype=dtype
        )

    def forward(
        self, x: Int[Tensor, " batch_size sequence_length"]
    ) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)
        x_normed = self.ln_final(x)
        output = self.lm_head(x_normed)
        return output


# 缩放点积注意力函数
def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    # 交换K的最后两维
    K_t = K.transpose(-2, -1)

    d_k = Q.shape[-1]
    scores = einsum(Q, K_t, "... queries d_k, ... d_k keys -> ... queries keys")
    scores = scores / math.sqrt(d_k)
    # mask按位取反，如果是false，则用inf替换。
    # masked_fill(condition, value)
    scores.masked_fill_(~mask, float("-inf"))
    # 对最后一维(keys) 进行softmax
    attn_probs = softmax(scores, dim=-1)

    attn_output = einsum(
        attn_probs, V, " ... queries keys, ... keys d_v -> ... queries d_v"
    )
    return attn_output
