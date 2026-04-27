from __future__ import annotations

# from cgitb import text
from heapq import merge
import os
from collections.abc import Iterable
from random import vonmisesvariate
from selectors import DefaultSelector
from sys import exception
from time import timezone
import tokenize
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

import regex as re
import json

# byte -> unicode_char 的表
from tests.common import gpt2_bytes_to_unicode


class Tokenizer:
    # 初始化tokenizer对象
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        # 把special_tokens从长到短排序，方便后续贪心匹配
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None
        # vocab的反查链表
        self.bytes_to_id = {}
        for key, value in vocab.items():
            self.bytes_to_id[value] = key

    # 根据文件初始化
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 读取的vocab: token_str -> token_id, 这里token_str并非直接的bytes转变的字符串，
        # 而是通过gpt2的方式映射后的str，目的是让所有byte都可打印可存文件
        # 因此需要先转置vocab，同时把str里每个字符通过byte_decoder转换为实际byte对应的int，再组合成bytes填入vocab
        # merges: str -> str, 同理
        # 读取文件
        with open(vocab_filepath) as vocab_f:
            vocab = json.load(vocab_f)
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                cleaned_line = line.rstrip()  # 删除末尾换行符和空白字符
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple(cleaned_line.split(" ")))

        # 构建 unicode-> byte 转换器
        byte_encoder = gpt2_bytes_to_unicode()
        byte_decoder = {}  # decoder即 encoder的转置
        for byte_value, unicode_char in byte_encoder.items():
            byte_decoder[unicode_char] = byte_value
        # vocab转置并恢复原始 bytes
        vocab_t = {}
        for vocab_item, vocab_index in vocab.items():
            byte_list = []
            for char in vocab_item:
                byte_list.append(byte_decoder[char])
            vocab_t[vocab_index] = bytes(byte_list)
        vocab = vocab_t
        # 确保所有special_tokens都在vocab内
        # vocab是dict[int,bytes]
        if special_tokens:
            for special_token in special_tokens:
                byte_endcoded_special_token = special_token.encode("utf-8")
                if byte_endcoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_endcoded_special_token
            pass
        # merges内容恢复原始bytes
        merges_t = []
        for merge_token1, merge_token2 in merges:
            merges_t.append(
                (
                    bytes([byte_decoder[token] for token in merge_token1]),
                    bytes([byte_decoder[token] for token in merge_token2]),
                )
            )
        merges = merges_t
        # for k, v in list(vocab.items())[:90]:
        #     print(k, v)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # pre-tokenizer预分词
        # 处理special_tokens，分
        if self.special_tokens:
            escaped_tokens = []
            for token in self.special_tokens:
                escaped_token = re.escape(token)
                escaped_tokens.append(escaped_token)
            pattern = "(" + "|".join(escaped_tokens) + ")"
            parts = re.split(pattern, text)
        else:
            self.special_tokens = []
            parts = [text]
        # print(f"special_tokens处理后: {parts}")
        # 清除空格
        filtered_parts = []
        for part in parts:
            if part:
                filtered_parts.append(part)
        parts = filtered_parts

        # 分流普通文本和special_tokens文本，获得对应 token_id
        ids = []
        for part in parts:
            if part not in self.special_tokens:
                # 按gpt-2的正则表达式分词
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                part = re.findall(PAT, part)
                # print(f"part = {part}")

                part_bytes = []
                # 转换为bytes串
                for pre_token in part:
                    token_bytes = pre_token.encode("utf-8")
                    token_seq = tuple(bytes([b]) for b in token_bytes)
                    part_bytes.append(token_seq)
                # print(f"part_bytes 结果:\n{part_bytes}")

                # 对预分词结果进行合并操作
                # 依照每一个merge，遍历一次part_bytes进行合并
                # print(f"merges长度: {len(self.merges)}")
                for merge in self.merges:
                    # print(f"第{epoch}轮合并：")
                    merged_tokens = []
                    for seq in part_bytes:
                        # print(token_seq)
                        new_seq = []
                        i = 0
                        seq_len = len(seq)
                        while i < seq_len:
                            if i < seq_len - 1 and (seq[i], seq[i + 1]) == merge:
                                token = seq[i] + seq[i + 1]
                                new_seq.append(token)
                                i += 2
                            else:
                                new_seq.append(seq[i])
                                i += 1
                        new_seq = tuple(new_seq)
                        merged_tokens.append(new_seq)
                    part_bytes = merged_tokens
                merged_tokens = part_bytes
                # print(merged_tokens)

                for token_sequence in merged_tokens:
                    for token_bytes in token_sequence:
                        ids.append(self.bytes_to_id[token_bytes])
                # print()
            else:
                # print(f"special类型: {type(part)}")
                ids.append(self.bytes_to_id[part.encode("utf-8")])
                # print()
        # print(ids)
        return ids

    # 返回值是Iterable，所以得用yield一点点返回
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            ids = self.encode(text)
            for token_id in ids:
                yield token_id

    # ids解码为string
    def decode(self, ids: list[int]) -> str:
        token_bytes_list = []
        for token_id in ids:
            token_bytes = self.vocab[token_id]
            token_bytes_list.append(token_bytes)
        decoded_bytes = b"".join(token_bytes_list)
        # 某些单独 token 的 bytes 不一定构成合法 UTF-8，使用 replace 避免 decode 时报错
        decoded_text = decoded_bytes.decode("utf-8", errors="replace")
        # print(decoded_text)
        return decoded_text

    # 测试用，将string转换为vocab内的token_id
    def give_ids(self, text) -> list[int]:
        ids = []
        for target in text:
            target = target.encode("utf-8")
            # print(f"finding: {target}")
            for token_id, token_bytes in self.vocab.items():
                if token_bytes == target:
                    ids.append(token_id)
        return ids


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

    raise NotImplementedError


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

    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_model"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_model"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


# 返回一个实现AdamW的torch.optim.Optimizer。
def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

    # 读取训练文件，当作一整个字符串
    print(f"[读入文件]\n文件路径：{input_path}")
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到文件: {input_path}") from None
    except Exception as e:
        raise Exception(f"异常错误") from None

    print(f"文本长度: {len(text)}")
    print()

    # 按special_tokens切分，过滤掉空字符串，返回parts列表
    print(f"[预分词]")
    print(f"special_token: {special_tokens}")
    if special_tokens:
        escaped_tokens = []
        for token in special_tokens:
            escaped_token = re.escape(token)
            escaped_tokens.append(escaped_token)
        pattern = "|".join(escaped_tokens)
        parts = re.split(pattern, text)
    else:
        parts = text
    filtered_parts = []
    for part in parts:
        if part:  # part不为空时
            filtered_parts.append(part)
    parts = filtered_parts

    # 按照gpt-2的正则表达式分词
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    temps = []
    for temp in parts:
        temps.extend(re.findall(PAT, temp))
    parts = temps

    # pretoken_counts记录tuple形式的bytes串和对应的频次
    pretoken_counts = {}
    for token in parts:
        token_bytes = token.encode("utf-8")
        token_tuple = tuple(bytes([b]) for b in token_bytes)
        pretoken_counts[token_tuple] = pretoken_counts.get(token_tuple, 0) + 1
    print("前5条pretoken_counts信息:")
    for i, (key, value) in enumerate(list(pretoken_counts.items())[:5], start=1):
        print(f"  {i:02d} {key}: {value}")
    print()

    # 初始化vocab和merge，内容为 256token + special
    print(f"[初始化词表]")
    vocab = {}
    merges = []
    for token_id in range(256):
        vocab[token_id] = bytes([token_id])
    for token_str in special_tokens:
        vocab[len(vocab)] = token_str.encode("utf-8")
    print(f"词表长度: {len(vocab)}")
    print("打印部分special_tokens: ")
    for token_id, token in list(vocab.items())[256:260]:
        print(f"  {token_id}: {token}")
    print()

    # merge的native实现
    # 循环直至达到vocab_size
    print(f"[merge阶段]")
    print(f"vocab_siz = {vocab_size}")
    while len(vocab) < vocab_size:
        print(f"词表长度: {len(vocab)}")

        # 计算出相邻序列的频次pair_count
        pair_count = {}
        for seq, freq in pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                pair_count[pair] = pair_count.get(pair, 0) + freq
        # print(f"  打印部分pair_count:")
        # for key, value in list(pair_count.items())[:5]:
        #     print(f"    {key}: {value}")

        # 计算频次最高的pair，频次相同则按字典序排序
        best_pair, best_count = max(
            pair_count.items(), key=lambda item: (item[1], item[0])
        )
        # 更新merges和vocab
        merges.append(best_pair)
        token_id = len(vocab)
        vocab[token_id] = best_pair[0] + best_pair[1]
        print(f"  best_pair: {best_pair}->{best_count}")
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
            new_pretoken_counts[new_seq] = new_pretoken_counts.get(new_seq, 0) + freq
        pretoken_counts = new_pretoken_counts
    return vocab, merges


# TODO: 按 special token 边界切分语料，并行执行 pre-tokenization
# TODO: 每轮 merge 后只更新受影响的 pair_count，避免重复遍历全部 pretoken_counts
# TODO: 测试并优化在TinyStories和OpenWebText数据集上的训练

# 测试单独函数效果
if __name__ == "__main__":
    vocab_path = "tests/fixtures/gpt2_vocab.json"
    merges_path = "tests/fixtures/gpt2_merges.txt"
    special_tokens = ["<|endoftext|>", "<|pad|>"]

    tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    # ids = tok.give_ids("hello")
    # tok.decode(ids)
    text = "the c<|endoftext|>at ate"
    text_encode = tok.encode(text)
    text_decode = tok.decode(text_encode)
    print(text)
    print(text_decode)
    print(text == text_decode)
