import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, num_heads: int = 8, d_model: int = 512,
                 num_layers: int = 4, dropout: float = 0.2):
        super(Transformer, self).__init__()
        self.model_name = "stock_transformer"
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(in_features=1, d_model=self.d_model, use_cls=True)

        # importance embeddings
        self.encoder = Encoder(num_heads, self.d_model, self.num_layers, dropout)
        self.prediction_head = nn.Linear(self.d_model, 2)

    def forward(self, x):
        x = x.unsqueeze(2)
        bs, n, _ = x.size()
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.prediction_head(x[:, 0])
        state = x[:, 0]
        return state, x[:, 1]


class Encoder(nn.Module):
    def __init__(self, num_heads, d_model, enc_layers, dropout):
        super(Encoder, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.enc_layers = enc_layers

        modules = []
        for _ in range(self.enc_layers):
            modules.append(
                EncoderBlock(num_heads=self.num_heads, d_model=self.d_model, drop_rate=dropout))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x: Tensor):
        for block in self.module_list:
            x = block(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, num_heads, d_model, drop_rate=0.3):
        super(EncoderBlock, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.drop_rate = drop_rate

        self.sa = MultiAttentionNetwork(d_model=d_model,
                                        attention_dim=d_model,
                                        num_heads=num_heads,
                                        dropout=drop_rate)
        self.mlp = MLP(d_model=d_model, scale=4, dropout=drop_rate)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p=drop_rate)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x: Tensor):
        x1 = self.sa(x, x)
        x = self.norm1(self.dropout1(x1) + x)  # residual

        x1 = self.mlp(x)
        x = self.norm2(self.dropout2(x1) + x)  # residual

        return x


class MultiAttentionNetwork(nn.Module):
    def __init__(self, d_model, attention_dim, num_heads=8, dropout=0.2):
        super(MultiAttentionNetwork, self).__init__()
        # module parameters
        self.d_model = d_model
        self.attention_dim = attention_dim
        assert attention_dim % num_heads == 0
        self.head_dim = self.attention_dim // num_heads
        self.num_heads = num_heads
        self.scale = d_model ** -0.5

        # module layers
        # q k v layers
        self.q = nn.Linear(d_model, attention_dim)
        self.k = nn.Linear(d_model, attention_dim)
        self.v = nn.Linear(d_model, attention_dim)
        self.dropout = nn.Dropout(p=dropout)

        # self attention projection layer
        self.feature_projection = nn.Linear(attention_dim, d_model)

    def forward(self, x, y):
        """
        :param x: query seq (batch_size, N, d_model)
        :param y: key, value seq
        :return:
        """
        batch_size, N, _ = x.size()
        _, M, _ = y.size()

        q = self.q(x).view(batch_size, N, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        k = self.k(y).view(batch_size, M, self.num_heads, -1) \
            .permute(0, 2, 1, 3)
        v = self.v(y).view(batch_size, M, self.num_heads, -1) \
            .permute(0, 2, 1, 3)

        attention_score = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attention_weight = F.softmax(attention_score, dim=3)
        attention_weight = self.dropout(attention_weight)
        attention_output = torch.matmul(attention_weight, v). \
            permute(0, 2, 1, 3).contiguous().view(batch_size, N, -1)

        attention_output = self.feature_projection(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self, d_model, scale=4, dropout=0.2):
        super(MLP, self).__init__()
        # module parameters
        self.d_model = d_model
        self.scale = scale

        # module layers
        self.fc1 = nn.Linear(d_model, scale * d_model)
        self.fc2 = nn.Linear(scale * d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class Embedding(nn.Module):
    def __init__(self, in_features: int = 1, d_model: int = 512, use_cls=True):
        super(Embedding, self).__init__()
        # model info
        self.in_features = in_features
        self.d_model = d_model
        self.use_cls = use_cls

        # cls token
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros((1, 1, d_model)))
        # model layers
        self.close_transform = nn.Linear(in_features, d_model)
        self.feature_transform = nn.Linear(in_features, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout=0.2)

    def forward(self, x: Tensor):
        batch_size = x.size()[0]

        ft = self.feature_transform(x[:, 1:])
        close = self.close_transform(x[:, 0]).unsqueeze(1)
        x = torch.cat([close, ft], dim=1)
        if self.use_cls:
            cls_token = self.cls_token.expand((batch_size, 1, self.d_model))
            x = torch.cat([cls_token, x], dim=1)
        x = self.pos_embedding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 len: int = 35):
        super(PositionalEncoding, self).__init__()
        self.len = len
        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)
        self.pos_emb = nn.Parameter(torch.zeros((1, len, d_model)))

    def forward(self, token_embedding: Tensor):
        batch_size = token_embedding.size()[0]
        pos_emb = self.pos_emb.expand((batch_size, self.len, self.d_model))

        return self.dropout(token_embedding + pos_emb)
