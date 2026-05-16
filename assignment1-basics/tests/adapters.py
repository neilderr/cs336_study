from __future__ import annotations

# from cgitb import text


import numpy.typing as npt
import numpy as np

import torch
from jaxtyping import Bool, Float, Int

# byte -> unicode_char 的表
from tests.common import gpt2_bytes_to_unicode
from tqdm import tqdm
import time

import math
from einops import rearrange, einsum

from collections.abc import Callable, Iterable
from typing import Optional

# cs336_basics
from cs336_basics.tokenizer import *
from cs336_basics.model import *
from cs336_basics.optimizer import *
from cs336_basics.data import *
from cs336_basics.nn_utils import *


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """

    # 这里weights是指定的权重，这里不需要我们初始化了
    # in_features实际是输入的x。d_in, d_out才是构造 Linear的 features
    linear = Linear(d_in, d_out)
    # 调用load_state_dict()覆盖weight，函数接收字典
    linear.load_state_dict({"weight": weights})
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    embedding = Embedding(vocab_size, d_model)
    embedding.load_state_dict({"weight": weights})
    return embedding(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a SwiGLU network, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Dimensionality of the up-project happening internally to your swiglu.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W1
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for W2
        w3_weight (Float[Tensor, "d_ff d_model"]): Stored weights for W3
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # You can also manually assign the weights
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight

    # 上面的注释暗示函数要用Linear实现 w1、w2、w3

    swiglu = SwiGLU(d_model, d_ff)
    # 注意这里w1是Linear，所以key得用w1.weight"
    swiglu.load_state_dict(
        {"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight}
    )
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        # max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi = MultiheadSelfAttention(d_model, num_heads)
    qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    multi.load_state_dict(
        {
            "qkv_proj.weight": qkv_proj_weight,
            "o_proj.weight": o_proj_weight,
        }
    )
    return multi(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This version of MHA should include RoPE.
    In this case, the RoPE embedding dimension must be the head embedding dimension (d_model // num_heads).
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.
        token_positions (Int[Tensor, " ... sequence_length"] | None): Optional tensor with the positions of the tokens

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    multi = MultiheadSelfAttention(
        d_model, num_heads, theta=theta, max_seq_len=max_seq_len
    )
    qkv_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
    multi.load_state_dict(
        {
            "qkv_proj.weight": qkv_proj_weight,
            "o_proj.weight": o_proj_weight,
        }
    )
    return multi.forward(in_features, is_rope=True, token_positions=token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    Run RoPE for a given input tensor.

    Args:
        d_k (int): Embedding dimension size for the query or key tensor.
        theta (float): RoPE parameter.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        in_query_or_key (Float[Tensor, "... sequence_length d_k"]): Input tensor to run RoPE on.
        token_positions (Int[Tensor, "... sequence_length"]): Tensor of shape (batch_size, sequence_length) with the token positions
    Returns:
        Float[Tensor, " ... sequence_length d_k"]: Tensor with RoPEd input.
    """
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use RoPE.
    Depending on your implementation, you may simply need to pass the relevant args
    to your TransformerBlock constructor, or you may need to initialize your own RoPE
    class and pass that instead.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        theta (float): RoPE parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features while using RoPE.
    """
    transformer_block = TransformerBlock(
        d_model,
        num_heads,
        d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    qkv_proj = torch.cat(
        [
            weights["attn.q_proj.weight"],
            weights["attn.k_proj.weight"],
            weights["attn.v_proj.weight"],
        ],
        dim=0,
    )
    transformer_block.load_state_dict(
        {
            "rms_norm_1.weight": weights["ln1.weight"],
            "attn.o_proj.weight": weights["attn.output_proj.weight"],
            "attn.qkv_proj.weight": qkv_proj,
            "rms_norm_2.weight": weights["ln2.weight"],
            "ffn.w1.weight": weights["ffn.w1.weight"],
            "ffn.w2.weight": weights["ffn.w2.weight"],
            "ffn.w3.weight": weights["ffn.w3.weight"],
        }
    )
    return transformer_block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    This function should use RoPE.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """

    transformer_lm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )

    # 构建权重字典
    full_state_dict = {
        "embedding.weight": weights["token_embeddings.weight"],
        "ln_final.weight": weights["ln_final.weight"],
        "lm_head.weight": weights["lm_head.weight"],
    }
    for i in range(num_layers):
        qkv_proj = torch.cat(
            [
                weights[f"layers.{i}.attn.q_proj.weight"],
                weights[f"layers.{i}.attn.k_proj.weight"],
                weights[f"layers.{i}.attn.v_proj.weight"],
            ],
            dim=0,
        )

        full_state_dict.update(
            {
                f"blocks.{i}.rms_norm_1.weight": weights[f"layers.{i}.ln1.weight"],
                f"blocks.{i}.rms_norm_2.weight": weights[f"layers.{i}.ln2.weight"],
                f"blocks.{i}.ffn.w1.weight": weights[f"layers.{i}.ffn.w1.weight"],
                f"blocks.{i}.ffn.w2.weight": weights[f"layers.{i}.ffn.w2.weight"],
                f"blocks.{i}.ffn.w3.weight": weights[f"layers.{i}.ffn.w3.weight"],
                f"blocks.{i}.attn.qkv_proj.weight": qkv_proj,
                f"blocks.{i}.attn.o_proj.weight": weights[
                    f"layers.{i}.attn.output_proj.weight"
                ],
            }
        )

    for k, v in full_state_dict.items():
        print(k, v)
    transformer_lm.load_state_dict(full_state_dict)
    return transformer_lm(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a RMSNorm affine transform,
    return the output of running RMSNorm on the input features.

    Args:
        d_model (int): The dimensionality of the RMSNorm input.
        eps: (float): A value added to the denominator for numerical stability.
        weights (Float[Tensor, "d_model"]): RMSNorm weights.
        in_features (Float[Tensor, "... d_model"]): Input features to run RMSNorm on. Can have arbitrary leading
            dimensions.

    Returns:
        Float[Tensor,"... d_model"]: Tensor of with the same shape as `in_features` with the output of running
        RMSNorm of the `in_features`.
    """
    rms_norm = RMSNorm(d_model, eps)
    rms_norm.load_state_dict({"weight": weights})
    return rms_norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """Given a tensor of inputs, return the output of applying SiLU
    to each element.

    Args:
        in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

    Returns:
        Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
        SiLU to each element.
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    return data_loader(dataset, batch_size, context_length, device=device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    return cross_entropy(inputs, targets)


# 给定一组参数，将它们的组合梯度裁剪为l2范数，最多为max_l2_norm。
def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """
    gradient_clipping(parameters, max_l2_norm)


# 返回一个实现AdamW类，并非实例
def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamW


# 给定余弦学习率衰减计划（带线性预热）的参数和迭代次数，返回给定迭代次数下的学习率。
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    return learning_rate_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


# 给定模型、优化器和迭代次数，将它们序列化到磁盘。
def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    save_checkpoint(model, optimizer, iteration, out)


# 给定序列化的检查点（路径或文件-like 对象），将序列化的状态恢复到给定的模型和优化器中。
# 返回我们在检查点中之前序列化的迭代次数。
def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    return load_checkpoint(src, model, optimizer)


# 给定词汇表、合并列表和特殊令牌列表，返回一个使用这些词汇表、合并和特殊令牌的BPE分词器
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """

    return Tokenizer(vocab, merges, special_tokens)


# 给定输入语料库的路径，训练一个BPE分词器，并输出其词汇表和合并规则。
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    #这里vocab_size是目标词表大小，包括special_tokens，而不是传入时vocab的大小

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.

    vocab = 字典里收录了哪些词
    merges = 学这些词时形成的优先级规则

    1、读取语料文件
    2、按 special tokens 做硬切分。这些 special tokens 自己不能参与 merge 统计
    3、对普通文本片段做 pre-tokenization。也就是把文本拆成适合 BPE 统计的更小单元
    4、从字节级初始词表开始，反复统计 pair，选最频繁 pair 做 merge
    5、维护：
    vocab：最后有哪些 token
    merges：每一步 merge 的顺序记录
    6、达到目标词表大小后返回 (vocab, merges)
    """
    # 从kwargs里取出合并模式，默认用增量方式
    merge_mode = kwargs.get("merge_mode", "incremental")

    # 启动计时器
    start_time = time.time()
    phase_start_time = start_time

    # 读取训练语料
    print(f"[读入文件 & 并行预分词]\n文件路径：{input_path}")
    pretoken_counts = {}
    # 并行化参数
    num_workers = kwargs.get("num_workers", 4)
    num_chunks = kwargs.get("num_chunks", num_workers * 8)
    mini_chunk_size = kwargs.get("mini_chunk_size", 4096)

    print(f"num_workers = {num_workers}")
    print(f"num_chunks = {num_chunks}")
    print(f"mini_chunk_size = {mini_chunk_size}")

    # 二进制打开文件，按照special_tokens切分文件
    # 返回一串边界，如 [0, 534000123, 1067000456, file_size]
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            b"<|endoftext|>",
            mini_chunk_size,
        )

    tasks = []
    # 相邻边界组合成区间
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((str(input_path), start, end, special_tokens))
    print(f"实际 chunk 数 = {len(tasks)}")

    progress_bar = tqdm(total=len(tasks), desc="Pretokenizing Chunks")

    with Pool(processes=num_workers) as pool:
        for local_counts in pool.imap_unordered(process_chunk, tasks):
            for token_tuple, count in local_counts.items():
                pretoken_counts[token_tuple] = (
                    pretoken_counts.get(token_tuple, 0) + count
                )
            progress_bar.update(1)
        progress_bar.close()

    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.2f} 秒")
    print()

    # 初始化vocab和merge，内容为 256token + special
    print(f"[初始化词表]")
    phase_start_time = time.time()
    vocab = {}
    merges = []
    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    for token_str in special_tokens:
        vocab[len(vocab)] = token_str.encode("utf-8")
    print(f"词表长度: {len(vocab)}")
    print(f"目标词表长度: {vocab_size}")
    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.6f} 秒")
    print()

    # merge阶段
    # 循环直至达到vocab_size
    print(f"[merge阶段]: {merge_mode} 模式")
    phase_start_time = time.time()
    print(f"vocab_size = {vocab_size}")
    # 进度条实现
    progress_bar = tqdm(total=vocab_size, desc="BEP merges", initial=len(vocab))
    # 增量法实现
    if merge_mode == "incremental":

        # 核心部分！！！
        # pair_to_seqs：某个pair出现在哪些seq里，每次获取best_pair后只需要修改其对应的seq
        # dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]，类似下面的结构
        """
        {
            (b't', b'h'): {
                (b't', b'h', b'e'),
                (b't', b'h', b'a', b't'),
            }
        }
        """
        # 开始完整遍历一次获取pair_count和pair_to_seqs
        pair_to_seqs = {}
        pair_count = {}
        for seq, freq in pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + freq
                # 如果这个pair没有对应的集合，就先创建一个新空集合
                if pair not in pair_to_seqs:
                    pair_to_seqs[pair] = set()
                # 把seq加入pair对应的集合里
                pair_to_seqs[pair].add(seq)

        # merge循环
        while len(vocab) < vocab_size:
            # 如果训练集太小，会出现没有可以merge的pair_count，此时直接退出
            if not pair_count:
                progress_bar.close()
                print(f"没有可以merge的pair，退出训练\n词表长度: {len(vocab)}\n")
                break
            # 计算频次最高的pair，频次相同则按字典序排序
            best_pair, best_count = max(
                pair_count.items(), key=lambda item: (item[1], item[0])
            )
            # 更新merges和vocab
            merges.append(best_pair)
            merged_pair = best_pair[0] + best_pair[1]
            token_id = len(vocab)
            vocab[token_id] = merged_pair

            # affected_seqs代表受影响的seqs
            # 如果best_pair存在，拿到它对应的seq。否则返回空集合set()
            # 转成list是方便后续更改
            affected_seqs = list(pair_to_seqs.get(best_pair, set()))

            for old_seq in affected_seqs:
                freq = pretoken_counts.pop(old_seq)
                old_seq_len = len(old_seq)
                # 1. 先把 old_seq 对所有 pair 的贡献删掉
                # 逐位置更新 pair_count
                old_pairs_in_seq = set()
                for i in range(old_seq_len - 1):
                    pair = (old_seq[i], old_seq[i + 1])
                    # 删除old_seq对pair_count的贡献
                    pair_count[pair] -= freq
                    if pair_count[pair] <= 0:
                        del pair_count[pair]
                    # 记录old_seq里出现过几种不同的pair
                    old_pairs_in_seq.add(pair)

                # 按“不同 pair”更新 pair_to_seqs
                for pair in old_pairs_in_seq:
                    # pair不在出现在old_seq里，因为old_seq将要删除
                    pair_to_seqs[pair].discard(old_seq)
                    if not pair_to_seqs[pair]:
                        del pair_to_seqs[pair]
                # 2. 构造 new_seq
                new_seq = []
                i = 0
                while i < old_seq_len:
                    if (
                        i < old_seq_len - 1
                        and (old_seq[i], old_seq[i + 1]) == best_pair
                    ):
                        new_seq.append(merged_pair)
                        i += 2
                    else:
                        new_seq.append(old_seq[i])
                        i += 1

                # 3. 把 new_seq 放回 pretoken_counts
                new_seq = tuple(new_seq)
                pretoken_counts[new_seq] = pretoken_counts.get(new_seq, 0) + freq
                # 4. 把 new_seq 对所有 pair 的贡献加回去
                for i in range(len(new_seq) - 1):
                    pair = (new_seq[i], new_seq[i + 1])
                    # 添加pair对pair_count的贡献
                    pair_count[pair] = pair_count.get(pair, 0) + freq
                    # pair出现在new_seq里
                    if pair not in pair_to_seqs:
                        pair_to_seqs[pair] = set()
                    pair_to_seqs[pair].add(new_seq)
            # 更新pretoken_counts和pair_counts
            progress_bar.update(1)

    # nativa实现
    else:
        while len(vocab) < vocab_size:
            # print(f"词表长度: {len(vocab)}")
            # 计算出相邻序列的频次pair_count
            pair_count = {}
            for seq, freq in pretoken_counts.items():
                for i in range(len(seq) - 1):
                    pair = (seq[i], seq[i + 1])
                    pair_count[pair] = pair_count.get(pair, 0) + freq
            # 如果训练集太小，会出现没有可以merge的pair_count，此时直接退出
            if not pair_count:
                progress_bar.close()
                print(f"没有可以merge的pair，退出训练\n词表长度: {len(vocab)}\n")
                break
            # 计算频次最高的pair，频次相同则按字典序排序
            best_pair, best_count = max(
                pair_count.items(), key=lambda item: (item[1], item[0])
            )
            # 更新merges和vocab
            merges.append(best_pair)
            token_id = len(vocab)
            vocab[token_id] = best_pair[0] + best_pair[1]
            # 更新pair_count
            new_pretoken_counts = {}
            for seq, freq in pretoken_counts.items():
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                        token = seq[i] + seq[i + 1]
                        new_seq.append(token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_seq = tuple(new_seq)
                new_pretoken_counts[new_seq] = (
                    new_pretoken_counts.get(new_seq, 0) + freq
                )
            pretoken_counts = new_pretoken_counts
            progress_bar.update(1)

    progress_bar.close()

    # 打印时间和内存
    phase_end_time = time.time()
    print(f"消耗时间: {phase_end_time-phase_start_time:.2f} 秒")
    print()
    print(f"总训练耗时: {time.time()-start_time:.2f} 秒")

    peak_memory_mb = get_peak_memory_mb()
    print(f"训练峰值内存: {peak_memory_mb:.2f} MB")

    # 打印最长token
    longest_token_id, longest_token_bytes = max(
        vocab.items(), key=lambda item: len(item[1])
    )
    print()
    print(f"最长 token id: {longest_token_id}")
    print(f"最长 token bytes 长度: {len(longest_token_bytes)}")
    byte_encoder = gpt2_bytes_to_unicode()
    longest_token_str = "".join(byte_encoder[b] for b in longest_token_bytes)
    print(f"最长 token: {longest_token_str}")
    return vocab, merges


# 测试单独函数效果
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=1e1)

    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()
