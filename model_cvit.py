import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from colorama import init, AnsiToWin32
import sys
from matplotlib import pyplot as plt
import de_Animator as Animator
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Conv_block(nn.Module):
    '''输入双通道（64*2*64*64）数据，输出嵌入好的图像块batch*num_steps-1*num_hiddens'''
    def __init__(self, patch_size, num_hiddens, **kwargs):

        super(Conv_block, self).__init__()
        self.net = nn.Sequential(
                    nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
                    nn.LayerNorm([32,32]),
                    nn.ReLU())
        self.proj = nn.Conv2d(16, num_hiddens, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(num_hiddens)
        self.relu = nn.ReLU()

    def forward(self, X ):
        X = X[:, :2, :, :]
        X = self.net(X)
        X = self.proj(X).flatten(2).permute(0, 2, 1)
        X = self.norm(X)
        X = self.relu(X)
        return X


class Class_token(nn.Module):
    '''添加分类向量'''

    def __init__(self, num_hiddens ):
        super(Class_token, self).__init__()
        self.num_hiddens = num_hiddens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.num_hiddens))

    def forward(self, X ):
        '''learnable parameters = class token'''
        cls_tokens = self.cls_token.expand(X.shape[0], -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)
        return X


class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough `P`
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.
    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout ):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    # Shape of `valid_lens`: (`batch_size`,) or (`batch_size`, no. of queries)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.
    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.
    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)
    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


class MultiHeadAttention(nn.Module):
    """Multi-head attention.
    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)
        # Shape of `valid_lens`:
        # (`batch_size`,) or (`batch_size`, no. of queries)
        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        output = self.attention(queries, keys, values)
        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class EncoderBlock(nn.Module):
    """Transformer encoder block.
    Defined in :numref:`sec_transformer`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X ):
        Y = self.addnorm1(X, self.attention(X, X, X ))
        return self.addnorm2(Y, self.ffn(Y))


class AddNorm(nn.Module):
    """Residual connection followed by layer normalization.

    Defined in :numref:`sec_transformer`"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network.
    Defined in :numref:`sec_transformer`"""

    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class Class_token_mlp(nn.Module):
    '''the mlp of class_token(X[:,0,:])'''

    def __init__(self, num_hiddens, mlp_hiddens, mlp_outs,
                 **kwargs):
        super(Class_token_mlp, self).__init__(**kwargs)
        self.dense1 = nn.Linear(num_hiddens, mlp_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(mlp_hiddens, mlp_outs)

    def forward(self, X):
        X = self.dense2(self.relu(self.dense1(X)))
        return X


def train_batch_ch13(net, X, y, loss, trainer, device):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13).
    Defined in :numref:`sec_image_augmentation`"""

    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(device) for x in X]
    else:
        X = X.to(device)
    y = y.to(device)
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),cmap='Reds'):
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
    sharex = True, sharey = True, squeeze=False)

    #sharex表示共享X轴,axes表示子图对象
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])

    fig.colorbar(pcm, ax=axes, shrink=0.6)
    # plt.savefig("C:/Users/Administrator/Desktop/vision-trans/heatmap.svg", dpi=600)
    d2l.plt.show()



