from abc import ABC
import numpy as np
import copy
from code.state.Attention import *


class SAKT(nn.Module, ABC):
    def __init__(self, args, device):
        super(SAKT, self).__init__()
        self.device = device
        embed_size = args.emb_dim
        self.encode_pos = False
        self.embed_size = embed_size
        self.attention_mode = args.attention_mode

        self.pos_emb = CosinePositionalEmbedding(embed_size, args.max_seq_len - 1)

        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(self.embed_size, args.num_heads, args.drop_prob,
                                                      self.attention_mode, self.device), args.num_attn_layers)

        self.dropout = nn.Dropout(p=args.drop_prob)
        # self.lin_out = nn.Linear(embed_size, 1)

    def forward(self, next_emb, ques_emb, interact_emb):
        inputs = F.relu(self.lin_in(torch.transpose(interact_emb, 0, 1)))  # [batch_size, seq_len, ques_dim]
        query = torch.transpose(next_emb, 0, 1)  # [batch_size, seq_len, ques_dim]

        inputs = inputs + self.pos_emb(inputs)
        # print(inputs.shape)
        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        outputs = self.attn_layers[0](query, inputs, inputs, mask)

        for l in self.attn_layers[1:]:
            residual = l(query, outputs, outputs, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return torch.transpose(outputs, 0, 1)  # [seq_len, batch_size, ques_dim]


class MultiHeadedAttention(nn.Module, ABC):
    def __init__(self, total_size, num_heads, drop_prob, attention_mode, device):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.seq_len = 199

        self.linear_layers = clone(nn.Linear(total_size, total_size), 3)
        self.dropout = nn.Dropout(p=drop_prob)
        self.prob_attn = None

        self.attention_mode = attention_mode
        assert self.attention_mode in ['dot', 'general']
        if self.attention_mode == 'dot':
            self.attention = dot_attention()
        else:
            self.attention = general_attention(self.head_size)

        # self.layer_norm = nn.LayerNorm(total_size)
        self.ffn = feedforward(total_size, drop_prob * 4)
        self.use_ffn = True
        self.dropout1 = nn.Dropout(p=drop_prob * 4)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length = query.shape[:2]
        input = query

        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)

        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # Apply attention
        # Apply attention
        if self.attention_mode == 'topk':
            out, self.prob_attn = self.attention(query, key, value, mask, self.k)
        else:
            out, self.prob_attn = self.attention(query, key, value, mask, self.dropout)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        # out = self.layer_norm(self.dropout1(out) + input)
        out = self.dropout1(out) + input
        if self.use_ffn:
            out = self.ffn(out)

        return out


class feedforward(nn.Module, ABC):
    def __init__(self, d_model, dropout):
        super().__init__()
        # self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, input):
        FFN1 = self.dropout1(self.activation(self.linear1(input)))
        FFN2 = self.dropout2(self.linear2(FFN1))
        out = FFN2 + input
        # return self.layer_norm(out)
        return out


class CosinePositionalEmbedding(nn.Module, ABC):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]  # ( 1,seq,  Feature)


def future_mask(seq_length):
    mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])
