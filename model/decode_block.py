import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, LeakyReLU, Dropout, ReLU, PReLU
import torch.nn.functional as F


class DecodeBlock(Module):

    def __init__(self, in_dim: int, out_dim: int, dropout=0.1):
        super().__init__()
        self.linear1 = Linear(in_dim, out_dim)
        self.leaky_relu = LeakyReLU(inplace=True)
        self.linear2 = Linear(out_dim, out_dim)
        if in_dim != out_dim:
            self.skip = nn.Sequential(Linear(in_dim, in_dim // 2),
                                      LeakyReLU(inplace=True),
                                      Linear(in_dim // 2, in_dim // 2),
                                      LeakyReLU(inplace=True),
                                      Linear(in_dim // 2, out_dim))
        else:
            self.skip = None
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.linear1(x)
        out = self.leaky_relu(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return self.dropout2(out)


class BilinearFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(BilinearFeatureFusion, self).__init__()
        self.bilinear = nn.Bilinear(input_dim, input_dim, input_dim)  # 双线性层
        self.fc1 = nn.Linear(input_dim * 3, input_dim)  # 输出降维
        self.act1 = PReLU()
        self.drop1 = Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        # 双线性交互
        interaction = self.bilinear(x1, x2)  # [batch_size, input_dim]

        # 拼接原始特征和交互特征
        concatenated_features = torch.cat([x1, x2, interaction], dim=-1)  # [batch_size, input_dim*3]

        # 降维到原始特征维度
        fused_feature = self.drop1(self.act1(self.fc1(concatenated_features)))  # [batch_size, input_dim]
        fused_feature = self.fc2(fused_feature)
        return fused_feature


class GatedFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(GatedFeatureFusion, self).__init__()
        self.gate_fc = nn.Linear(input_dim * 2, 1)  # 门控网络
        self.fc1 = nn.Linear(input_dim, input_dim)  # 输出降维
        self.act1 = PReLU()
        self.drop1 = Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        # 将两个特征拼接
        concatenated_features = torch.cat([x1, x2], dim=-1)  # [batch_size, input_dim*2]

        # 计算门控权重
        gate = torch.sigmoid(self.gate_fc(concatenated_features))  # [batch_size, 1]

        # 门控加权融合
        fused_feature = gate * x1 + (1 - gate) * x2  # [batch_size, input_dim]
        fused_feature = self.drop1(self.act1(self.fc1(fused_feature)))  # [batch_size, input_dim]
        fused_feature = self.fc2(fused_feature)
        return fused_feature


class DynamicFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(DynamicFeatureFusion, self).__init__()
        self.weight_fc = nn.Linear(input_dim * 2, 2)  # 动态权重生成
        self.fc1 = nn.Linear(input_dim, input_dim)  # 输出降维
        self.act1 = PReLU()
        self.drop1 = Dropout(dropout)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        # 拼接两个特征
        concatenated_features = torch.cat([x1, x2], dim=-1)  # [batch_size, input_dim*2]

        # 计算动态权重
        weights = F.softmax(self.weight_fc(concatenated_features), dim=-1)  # [batch_size, 2]

        # 动态加权融合
        fused_feature = weights[:, 0:1] * x1 + weights[:, 1:2] * x2  # [batch_size, input_dim]
        fused_feature = self.drop1(self.act1(self.fc1(fused_feature)))  # [batch_size, input_dim]
        fused_feature = self.fc2(fused_feature)
        return fused_feature


class CatFeatureFusion(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0):
        super(CatFeatureFusion, self).__init__()
        self.ff_block = nn.Sequential(
            Linear(2 * input_dim, input_dim),
            ReLU(),
            Linear(input_dim, input_dim),
            ReLU(),
            Linear(input_dim, output_dim)
        )

    def forward(self, x1, x2):
        concatenated_features = torch.cat([x1, x2], dim=-1)
        return self.ff_block(concatenated_features)
