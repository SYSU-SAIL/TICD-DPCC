import torch
from torch import nn, Tensor
from torch.nn import functional as F, Linear, Dropout, LayerNorm, ReLU, ModuleList
import copy


class TemporalTransformerLayer(nn.Module):
    def __init__(self, embedding_size: int, nhead: int, dim_feedforward: int = 2048, drop_out=0.1,
                 layer_norm_eps: float = 1e-5, bias: bool = True):
        super(TemporalTransformerLayer, self).__init__()
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.multihead_attn = nn.MultiheadAttention(self.embedding_size, self.nhead, dropout=drop_out, batch_first=True,
                                                    add_bias_kv=True)
        self.linear1 = Linear(embedding_size, dim_feedforward, bias=bias)
        self.dropout = Dropout(drop_out)
        self.linear2 = Linear(dim_feedforward, embedding_size, bias=bias)
        self.norm1 = LayerNorm(embedding_size, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(embedding_size, eps=layer_norm_eps, bias=bias)
        self.dropout1 = Dropout(drop_out)
        self.dropout2 = Dropout(drop_out)  # 拓展矩阵
        self.activation = ReLU()

    def forward(self, query_embedding: torch.Tensor, key_embedding: torch.Tensor, ref_embedding: torch.Tensor,
                key_padding_mask: torch.Tensor):
        """
        给定参考帧树状索引的祖父节点下所有父级节点，提供Tree Attention
        :param key_padding_mask: 用于标记未被占用的参考父节点
        :param key_embedding: 参考树祖父节点下所有父级节点嵌入 [batch_size, 1*8*8, embedding_size]
        :param ref_embedding: 参考树祖父节点下所有同级节点嵌入 [batch_size, 1*8*8*8, embedding_size]
        :param query_embedding: 当前树下所有父级别节点嵌入 [batch_size, 1, embedding_size]
        :return:
        """
        x = ref_embedding
        context = self.norm1(x + self._x_block(query_embedding, key_embedding, x, key_padding_mask))
        context = self.norm2(context + self._ff_block(context))
        return context

    # self-attention block
    def _x_block(self, q, k, x: Tensor, key_padding_mask) -> Tensor:
        x = self.multihead_attn(q, k, x, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TemporalTransformer(nn.Module):
    def __init__(self, embedding_size, nhead, num_layers: int = 3, dropout=0.2):
        super(TemporalTransformer, self).__init__()
        encoder_layer = TemporalTransformerLayer(embedding_size, nhead, drop_out=dropout)
        self.layers = ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, query_embedding: torch.Tensor, key_embedding: torch.Tensor, ref_embedding: torch.Tensor,
                key_padding_mask: torch.Tensor):
        output = ref_embedding
        for mod in self.layers:
            output = mod(query_embedding, key_embedding, output, key_padding_mask)
        return output
