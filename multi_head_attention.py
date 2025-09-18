import math
import torch
from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, num_heads, dropout,
                 bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention  = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
    
    def forward(self, queries, keys, values, valid_lens):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            # 这里是为了对齐上边的reshape将num_heads也复制出来
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output = transpose_output(output, self.num_heads)
        return self.W_o(output)


# 这里就是为了将最后一个维度拆成(num_heads, head_dim)然后将num_heads和batch_size合并在一起
def transpose_qkv(X, num_heads):
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

# 这里就是上边的逆操作将num_heads拆出来然后concat到一起 
def transpose_output(X, num_heads):
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)

class DotProductAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    
    def masked_softmax(self, X, valid_lens):
        if valid_lens is None:
            return F.softmax(X, dim=-1)
        else:
            shape = X.shape
            # 这里前边的None是将向量扩展成相同的行，后边的None是将向量扩展成相同的列
            mask = torch.arange(shape[-1], device=X.device)[None, :] < valid_lens[:, None]
            if mask.shape != X.shape:
                mask = mask.unsqueeze(1).repeat(1, 4, 1)
            X[~mask] = -1e6
            return F.softmax(X, dim=-1).reshape(shape)

    def forward(self, queries, keys, values, valid_lens):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
if __name__ == "__main__":
    # 验证一下这里的MultiHeadAttention
    num_hiddens, num_heads = 100, 5
    attention = MultiHeadAttention(100, 100, 100, num_hiddens, num_heads, 0.5)
    attention.eval()
    batch_size, num_queries = 2, 4
    num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
    queries = torch.ones((batch_size, num_queries, 100))
    keys = torch.ones((batch_size, num_kvpairs, 100))
    values = torch.ones((batch_size, num_kvpairs, 100))
    print(attention(queries, keys, values, valid_lens).shape)
    
