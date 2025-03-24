import math
import torch
from functorch.dim import Tensor
from torch import nn
from torch.nn import Embedding, Dropout, LayerNorm, TransformerEncoderLayer, TransformerEncoder, Linear, ReLU, PReLU, \
    TransformerDecoderLayer, TransformerDecoder, init
from config import config
from model.decode_block import DecodeBlock, BilinearFeatureFusion, GatedFeatureFusion, CatFeatureFusion


class RecursiveAttention(nn.Module):
    def __init__(self, token_size, output_size, attention_heads, ancestor_num, context_len, anchor_num, dropout=0.1):
        super(RecursiveAttention, self).__init__()
        self.embedding_size = (config.network.leaf_embedding_dim +
                               config.network.octant_embedding_dim + config.network.level_embedding_dim)  # 嵌入维度
        self.ancestor_num = ancestor_num
        self.context_len = context_len
        self.anchor_num = anchor_num
        self.hidden_size = (ancestor_num + 1) * self.embedding_size
        self.attention_heads = attention_heads
        self.num_hidden_layers = config.network.num_hidden_layers
        self.dim_feedforward = config.network.dim_feedforward
        anchor_step = context_len // anchor_num
        self.ancestral_index = list(range(0, context_len, anchor_step))
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ TOKEN相关 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.leaf_encoder = Embedding(token_size, config.network.leaf_embedding_dim, padding_idx=0)
        self.octant_encoder = Embedding(8, config.network.octant_embedding_dim)
        self.level_encoder = Embedding(config.network.max_octree_level, config.network.level_embedding_dim)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 构建EHEM预测模型 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        encoder_layer = TransformerEncoderLayer(self.hidden_size, self.attention_heads, self.dim_feedforward,
                                                dropout, 'gelu', batch_first=True, norm_first=True)
        self.init_transformer_layer(encoder_layer)
        self.temporal_encoder = TransformerEncoder(encoder_layer, self.num_hidden_layers)
        self.ancestral_encoder = TransformerEncoder(encoder_layer, self.num_hidden_layers)
        # self.sibling_unknown_encoder = TransformerEncoder(encoder_layer, self.num_hidden_layers)
        self.sibling_known_encoder = TransformerEncoder(encoder_layer, self.num_hidden_layers)

        decoder_layer = TransformerDecoderLayer(self.hidden_size, self.attention_heads, self.dim_feedforward, dropout,
                                                'gelu', batch_first=True, norm_first=True)
        self.init_transformer_layer(decoder_layer)
        self.ancestral_decoder = TransformerDecoder(decoder_layer, self.num_hidden_layers)
        self.sibling_decoder1 = TransformerDecoder(decoder_layer, self.num_hidden_layers)
        self.sibling_decoder2 = TransformerDecoder(decoder_layer, self.num_hidden_layers)

        self.ancestral_output_layer = CatFeatureFusion(self.hidden_size, output_size,dropout)
        self.sibling_output_layer = CatFeatureFusion(self.hidden_size, output_size,dropout)

        # self.ancestral_ff_block = nn.Sequential(
        #     Linear(self.hidden_size, self.hidden_size),
        #     PReLU(),
        #     Dropout(dropout),
        #     Linear(self.hidden_size, self.hidden_size),
        # )
        #
        # self.sibling_ff_block = nn.Sequential(
        #     Linear(self.hidden_size, self.hidden_size),
        #     PReLU(),
        #     Dropout(dropout),
        #     Linear(self.hidden_size, self.hidden_size),
        # )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 初始化权重 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # nn.init.xavier_normal_(self.leaf_encoder.weight)
        nn.init.xavier_normal_(self.octant_encoder.weight)
        nn.init.xavier_normal_(self.level_encoder.weight)

    def forward(self,
                context: Tensor,
                context_parent: Tensor,
                refer_context: Tensor,
                refer_parent: Tensor,
                label: Tensor):
        """
        预测模型
        :param context: 当前树上下文
        :param context_mask: 树信息上下文掩码
        :param context_parent: 上下文父节点信息
        :param parent_node: 父节点
        :param refer_context: 参考上下文
        :param refer_context_mask:
        :param refer_parent:
        :return:
        """
        N, L, P, _ = context.shape
        _, R, T, _ = refer_context.shape

        def get_context_embedded(context: torch.Tensor):
            # 针对空点有-1的情况进行+1处理
            leaf = context[:, :, :, 4]  # 针对[CLS][SEP] + 2
            level = context[:, :, :, 2]  # 针对空特征 +1 针对[CLS][SEP] + 2
            octant = context[:, :, :, 3]  # 针对空特征 +1 针对[CLS][SEP] + 2
            leaf_embedded = self.leaf_encoder(leaf.long())
            level_embedded = self.level_encoder(level.long())
            octant_embedded = self.octant_encoder(octant.long())
            context_embedded = torch.cat((leaf_embedded, level_embedded, octant_embedded), dim=-1)
            src = context_embedded.view(N, -1, self.hidden_size)
            return src

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 组装特征 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ancestral_idx = torch.cat((context_parent, context), dim=2)
        temporal_idx = torch.cat((refer_parent, refer_context), dim=2)
        sibling_known_context = context.clone()
        sibling_known_context[:, self.ancestral_index, 0, 4] = label[:, self.ancestral_index]
        sibling_known_idx = torch.cat((context_parent, sibling_known_context), dim=2)
        sibling_known_idx = sibling_known_idx[:, self.ancestral_index]
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 嵌入特征 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ancestral_embedded = get_context_embedded(ancestral_idx)
        sibling_known_embedded = get_context_embedded(sibling_known_idx)
        temporal_embedded = get_context_embedded(temporal_idx)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 空间上下文信息 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        temporal_embedded = self.temporal_encoder(temporal_embedded)
        ancestral_embedded = self.ancestral_encoder(ancestral_embedded)
        sibling_known_embedded = self.sibling_known_encoder(sibling_known_embedded)

        ancestral_output = self.ancestral_decoder(ancestral_embedded, temporal_embedded)
        ancestral_hat = self.ancestral_output_layer(ancestral_embedded[:, self.ancestral_index],
                                                    ancestral_output[:, self.ancestral_index])

        sibling_output1 = self.sibling_decoder1(ancestral_embedded, sibling_known_embedded)
        sibling_output2 = self.sibling_decoder2(ancestral_embedded, temporal_embedded)
        sibling_hat = self.sibling_output_layer(sibling_output1, sibling_output2)

        output = sibling_hat
        output[:, self.ancestral_index, :] = ancestral_hat
        return output

    def decode_with_ancestor(self, context: Tensor,
                             context_parent: Tensor,
                             refer_context: Tensor,
                             refer_parent: Tensor):
        """
        使用仅包含父节点信息的上下文进行解码
        :param context: **不包含** 兄弟节点的上下文信息
        :param context_parent: 父节点上下文信息
        :param refer_context: 参考上下文信息
        :param refer_parent: 参考父节点上下文
        :return: 解码对数概率
        """
        N, L, P, _ = context.shape
        _, R, T, _ = refer_context.shape

        def get_context_embedded(context: torch.Tensor):
            # 针对空点有-1的情况进行+1处理
            leaf = context[:, :, :, 4]  # 针对[CLS][SEP] + 2
            level = context[:, :, :, 2]  # 针对空特征 +1 针对[CLS][SEP] + 2
            octant = context[:, :, :, 3]  # 针对空特征 +1 针对[CLS][SEP] + 2
            leaf_embedded = self.leaf_encoder(leaf.long())
            level_embedded = self.level_encoder(level.long())
            octant_embedded = self.octant_encoder(octant.long())
            context_embedded = torch.cat((leaf_embedded, level_embedded, octant_embedded), dim=-1)
            src = context_embedded.view(N, -1, self.hidden_size)
            return src

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 组装特征 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ancestral_idx = torch.cat((context_parent, context), dim=2)
        temporal_idx = torch.cat((refer_parent, refer_context), dim=2)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 嵌入特征 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ancestral_embedded = get_context_embedded(ancestral_idx)
        temporal_embedded = get_context_embedded(temporal_idx)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 空间上下文信息 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        temporal_embedded = self.temporal_encoder(temporal_embedded)
        ancestral_embedded = self.ancestral_encoder(ancestral_embedded)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 时空上下文 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ancestral_output = self.ancestral_decoder(ancestral_embedded, temporal_embedded)
        ancestral_hat = self.ancestral_output_layer(ancestral_embedded[:, self.ancestral_index],
                                                    ancestral_output[:, self.ancestral_index])
        return ancestral_hat

    def decode_with_sibling(self, context: Tensor,
                            context_parent: Tensor,
                            refer_context: Tensor,
                            refer_parent: Tensor):
        """
        使用包含兄弟信息的上下文进行解码
        :param context: **包含** 兄弟信息的上下文
        :param context_parent: 父节点上下文
        :param refer_context: 参考上下文
        :param refer_parent: 参考父节点上下文
        :return: 解码对数概率
        """
        N, L, P, _ = context.shape
        _, R, T, _ = refer_context.shape

        def get_context_embedded(context: torch.Tensor):
            # 针对空点有-1的情况进行+1处理
            leaf = context[:, :, :, 4]  # 针对[CLS][SEP] + 2
            level = context[:, :, :, 2]  # 针对空特征 +1 针对[CLS][SEP] + 2
            octant = context[:, :, :, 3]  # 针对空特征 +1 针对[CLS][SEP] + 2
            leaf_embedded = self.leaf_encoder(leaf.long())
            level_embedded = self.level_encoder(level.long())
            octant_embedded = self.octant_encoder(octant.long())
            context_embedded = torch.cat((leaf_embedded, level_embedded, octant_embedded), dim=-1)
            src = context_embedded.view(N, -1, self.hidden_size)
            return src

        ancestral_context = context.clone()
        ancestral_context[:, self.ancestral_index, 0, 4] = 0
        ancestral_idx = torch.cat((context_parent, ancestral_context), dim=2)
        sibling_known_idx = torch.cat((context_parent, context), dim=2)
        sibling_known_idx = sibling_known_idx[:, self.ancestral_index]
        temporal_idx = torch.cat((refer_parent, refer_context), dim=2)

        ancestral_embedded = get_context_embedded(ancestral_idx)
        temporal_embedded = get_context_embedded(temporal_idx)
        sibling_known_embedded = get_context_embedded(sibling_known_idx)

        temporal_embedded = self.temporal_encoder(temporal_embedded)
        ancestral_embedded = self.ancestral_encoder(ancestral_embedded)
        sibling_known_embedded = self.sibling_known_encoder(sibling_known_embedded)

        sibling_output1 = self.sibling_decoder1(ancestral_embedded, sibling_known_embedded)
        sibling_output2 = self.sibling_decoder2(ancestral_embedded, temporal_embedded)
        sibling_hat = self.sibling_output_layer(sibling_output1, sibling_output2)
        return sibling_hat

    @staticmethod
    def init_transformer_layer(layer):
        """
        对 Transformer Encoder 和 Decoder Layer 进行统一初始化。
        该方法适用于 TransformerEncoderLayer 和 TransformerDecoderLayer。
        """
        # 初始化多头注意力层的权重
        if hasattr(layer, 'self_attn'):
            init.xavier_uniform_(layer.self_attn.in_proj_weight)  # 输入权重
            init.xavier_uniform_(layer.self_attn.out_proj.weight)  # 输出权重
            init.zeros_(layer.self_attn.in_proj_bias)
        if hasattr(layer, 'multihead_attn'):
            init.xavier_uniform_(layer.multihead_attn.in_proj_weight)  # 交叉注意力输入权重
            init.xavier_uniform_(layer.multihead_attn.out_proj.weight)  # 交叉注意力输出权重
            init.zeros_(layer.multihead_attn.in_proj_bias)
        # 初始化前馈网络的权重
        if hasattr(layer, 'linear1'):
            init.xavier_uniform_(layer.linear1.weight)  # 第一个前馈层
            init.zeros_(layer.linear1.bias)
        if hasattr(layer, 'linear2'):
            init.xavier_uniform_(layer.linear2.weight)  # 第二个前馈层
            init.zeros_(layer.linear2.bias)
        # 初始化 LayerNorm 层的权重和偏置
        if hasattr(layer, 'norm1'):
            init.ones_(layer.norm1.weight)
            init.zeros_(layer.norm1.bias)
        if hasattr(layer, 'norm2'):
            init.ones_(layer.norm2.weight)
            init.zeros_(layer.norm2.bias)
        if hasattr(layer, 'norm3'):
            init.ones_(layer.norm3.weight)
            init.zeros_(layer.norm3.bias)
