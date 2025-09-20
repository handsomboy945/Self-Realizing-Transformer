import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from multi_head_attention import MultiHeadAttention
from d2l import torch as d2l

class PositionWiseDDN(nn.Module):
    def __init__(self, ffn_num_inputs, hidden_size, ffn_num_output, **kwargs):
        super(PositionWiseDDN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_inputs, hidden_size)
        self.dense2 = nn.Linear(hidden_size, ffn_num_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
    
class AddNorm(nn.Module):
    # 这里需要传入的参数就是归一化的形状
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x, y):
        return self.ln(self.dropout(y) + x)
    
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, 
            num_hiddens, num_heads,
            dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseDDN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 这里应当先把残差作为第一个参数传进去进行dropout
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

# 这里根据sin与cos分别进行位置编码加入到X数据中
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.P = torch.zeros((1, int(max_len), int(num_hiddens)))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                                 EncoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))
    def forward(self, X, valid_lens, *args):
        # 这里是因为这里的embedding后的维度越大其值越小，所以这里乘以 math.sqrt(self.num_hiddens)将其放到-1到1这么一个范围
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    """解码器中第i个模块"""
    def __init__(self, key_size, query_size, value_size, num_hidden,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hidden, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hidden, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseDDN(ffn_num_input, ffn_num_hiddens, num_hidden)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        # state中有三个东西分别是encoder输出、valid_lens以及之前的key_value
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 这里就是在做t时刻的输出的时候要把前边所有时刻的输入全都concat起来
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            # 如果是训练阶段，这里需要将预时间步之前的全都掩码掉，这里生成合法序列
            batch_size, seq_len, _ = X.shape
            dec_valid_lens = torch.arange(1, seq_len + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 如果是预测阶段这里倒是不需要，因为训练阶段这里就是一个一个输入的所以这里不需要生成合法序列这里是什么就是什么
            dec_valid_lens = None
        # 这里用的是输入的X因此用的是dec_valid_lens
        X2 = self.attention1(X, X, X, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 这里是encoder的输出所以用的也是encoder的valid_lens
        X3 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, X3)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(
                key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens,
                num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    # @property装饰器可以将一个方法变成属性调用，这里将attention_weights变成一个私有变量是为了防止外部变量访问导致变量混乱
    @property
    def attention_weights(self):
        return self._attention_weights

if __name__ == "__main__":
    # 训练阶段
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]

    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

    encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
                                num_hiddens, norm_shape, ffn_num_input,
                                ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
                                num_hiddens, norm_shape, ffn_num_input,
                                ffn_num_hiddens, num_heads, num_layers, dropout)
    net = d2l.EncoderDecoder(encoder, decoder)
    net.decoder.training = True
    d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    # 预测阶段
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    net.decoder.training = False
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = d2l.predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
            f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
