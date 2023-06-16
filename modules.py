import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
from torch.autograd import Variable

class GraphAttentionLayer1(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, mu=0.001, concat=False):
        super(GraphAttentionLayer1, self).__init__()
        self.in_features = in_features  
        self.out_features = out_features 
        self.dropout = dropout 
        self.alpha = alpha 
        self.concat = concat
        self.mu = mu

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) 
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  

        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):#inp为(10,11,600)
        """
        inp: input_fea [Batch_size, N, in_features]
        """
        h = torch.matmul(inp, self.W)  # [batch_size, N, out_features]  h为(10,11,600) W(600,600)
        N = h.size()[1]  #11
        B = h.size()[0]  # B batch_size 10

        a = h[:, 0, :].unsqueeze(1).repeat(1, N, 1)  # [batch_size, N, out_features] (10,11,600)
        a_input = torch.cat((h, a), dim=2)  # [batch_size, N, 2*out_features] (10,11,1200)

        # a_input = torch.cat([h.repeat(1, 1, N).view(args.batch_size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(args.batch_size, N, -1, 2 * self.out_features)

        e = self.leakyrelu(torch.matmul(a_input, self.a)) #e(10,11,1) a_input(10,11,1200) self.a(1200,1)
        # [batch_size, N, 1] 

        attention = F.softmax(e, dim=1)  # [batch_size, N, 1]#(10,11,1)
        attention = attention - self.mu
        attention = (attention + abs(attention)) / 2.0 #(10,11,1)
        # print(attention)
        attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
        # print(attention)
        attention = attention.view(B, 1, N)#(10,1,11)
        h_prime = torch.matmul(attention, h).squeeze(1)  # [batch_size, 1, N]*[batch_size, N, out_features] => [batch_size, 1, out_features]
       #h_prime(10,600)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class AttentionSelectContext(nn.Module):  
    def __init__(self, dim, dropout=0.0, BiLSTM_hidden_size = 100, Bilstm_num_layers = 2, 
                 Bilstm_seq_length = 3, BiLSTM_input_size = 100, max_rel = 10, max_tail = 10):
        super(AttentionSelectContext, self).__init__()
        self.hidden_size = BiLSTM_hidden_size
        self.num_layers = Bilstm_num_layers
        self.seq_length = Bilstm_seq_length
        self.BiLSTM_input_size = BiLSTM_input_size
        self.num_neighbor = max_rel
        self.lstm = nn.LSTM(BiLSTM_input_size, BiLSTM_hidden_size, Bilstm_num_layers, batch_first=True, bidirectional=True)
        self.device = "cuda:0"
        self.attention = GraphAttentionLayer1(self.hidden_size * 2 * self.seq_length, self.hidden_size * 2 * self.seq_length, dropout=dropout, alpha=0.2, mu=0.005, concat=False)

    def forward(self, left, right, mask_left=None, mask_right=None):
        """
        :param left: (head, rel, tail)
        :param right:
        :param mask_right:
        :param mask_left:
        :return:
        """
        head_left, rel_left, tail_left = left # The shape of the head entity embedding is (5,100). The embeddings of neighbor relations are(5,10,100). Its one-hop neighbors are(5,10,100)
        head_right, rel_right, tail_right = right #(5,100) (5,10,100) (5,10,100)
        batch_size = head_left.shape[0] #5
        weak_rel_l = (head_right - head_left).unsqueeze(1)
        weak_rel_r = (head_left - head_right).unsqueeze(1)

        head_left = head_left.unsqueeze(1) #(5,1,100) the embeddings of the head entities of adjacent triples.
        head_right = head_right.unsqueeze(1) #(5,1,100) the embeddings of the tail entities of adjacent triples.
        head = torch.cat((head_left, head_right), dim=1) #(5,2,100) 
        head = head.view(-1, 100) #(10,100) 
        head = torch.repeat_interleave(head, self.num_neighbor + 1, dim=0) #(110,100) 

        # tail = torch.cat((head_right, head_left), dim=1)
        # tail = tail.view(-1, 100)
        # tail = torch.repeat_interleave(tail, 39 + 1, dim=0)

        tail_r = torch.cat((head_left, tail_right), dim=1) #(5,11,100)
        tail_l = torch.cat((head_right, tail_left), dim=1)  #(5,11,100)
        tail = torch.cat((tail_l, tail_r), dim=1).view(-1,100)#(110,100)

        rel_l = torch.cat((weak_rel_l, rel_left), dim=1)#(5,11,100)
        rel_r = torch.cat((weak_rel_r, rel_right), dim=1)#(5,11,100)
        relation = torch.cat((rel_l, rel_r), dim=1).view(-1,100)#(110,100)

        batch_triples_emb = torch.cat((head, relation), dim=1)#(110,200)
        batch_triples_emb = torch.cat((batch_triples_emb, tail), dim=1) #(110,300)
        x = batch_triples_emb.view(2*(self.num_neighbor + 1), -1, self.BiLSTM_input_size)#(110,3,100)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)# 2 for bidirection num_layers为2 hidden_size为100
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        #h0,c0为(4,110,100)
        out, _ = self.lstm(x, (h0, c0))#x(110,3,100)  h_0(4,110,100)  c_0(4,110,100) 

        out = out.reshape(-1, self.hidden_size * 2 * self.seq_length) #(110,600) seq_length is 3 
        out = out.reshape(-1, self.num_neighbor + 1, self.hidden_size * 2 * self.seq_length) #num_neighbor为10  #(10,11,600) 

        out_att = self.attention(out)#(10,600)

        out = out.reshape(-1, self.num_neighbor * 2 + 2, self.hidden_size * 2 * self.seq_length)#(5,22,600)

        out = out.reshape(batch_size, -1, 2 * 3 * self.hidden_size)#(5,22,600)
        out_att = out_att.reshape(batch_size, -1, 2 * 3 * self.hidden_size)#(5,2,600)

        pos_h = out[:, 0, :]#(5,600)
        pos_z0 = out_att[:, 0, :]#(5,600) 
        pos_z1 = out_att[:, 1, :]#(5,600) 

        return pos_h, pos_z0, pos_z1


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention
    """

    def __init__(self, attn_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        :param attn_mask: [batch, time]
        :param scale:
        :param q: [batch, time, dim]
        :param k: [batch, time, dim]
        :param v: [batch, time, dim]
        :return:
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attn = attn * scale
        if attn_mask:
            attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """ Implement without batch dim"""

    def __init__(self, model_dim, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        To be efficient, multi- attention is cal-ed in a matrix totally
        :param attn_mask:
        :param query: [batch, time, per_dim * num_heads]
        :param key:
        :param value:
        :return: [b, t, d*h]
        """
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attn = self.dot_product_attention(query, key, value, scale, attn_mask)
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)
        output = self.layer_norm(residual + output)
        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model], dtype=torch.float)
        position_encoding = torch.tensor(position_encoding, dtype=torch.float)

        position_encoding = torch.cat((pad_row, position_encoding), dim=0)

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, batch_len, seq_len):
        """
        :param batch_len: scalar
        :param seq_len: scalar
        :return: [batch, time, dim]
        """
        input_pos = torch.tensor([list(range(1, seq_len + 1)) for _ in range(batch_len)]).cuda()
        return self.position_encoding(input_pos)


class GELU(nn.Module):
    """
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.gelu = GELU()

    def forward(self, x):
        """

        :param x: [b, t, d*h]
        :return:
        """
        output = x.transpose(1, 2)  # [b, d*h, t]
        output = self.w2(self.gelu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        # ffn_dim
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)
        return output, attention


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim=100, ffn_dim=800, num_heads=4, dropout=0.1, num_layers=6, max_seq_len=3,
                 with_pos=True):
        super(TransformerEncoder, self).__init__()
        self.with_pos = with_pos
        self.num_heads = num_heads

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim * num_heads, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)
        self.rel_embed = nn.Parameter(torch.rand(1, model_dim), requires_grad=True)

    def repeat_dim(self, emb):
        """
        :param emb: [batch, t, dim]
        :return:
        """
        return emb.repeat(1, 1, self.num_heads)

    def forward(self, left, right):
        """
        :param left: [batch, dim]
        :param right: [batch, dim]
        :return:
        """
        batch_size = left.size(0)
        rel_embed = self.rel_embed.expand_as(left)

        left = left.unsqueeze(1)
        right = right.unsqueeze(1)
        rel_embed = rel_embed.unsqueeze(1)  # [batch, 1, dim]

        seq = torch.cat((left, rel_embed, right), dim=1)
        pos = self.pos_embedding(batch_len=batch_size, seq_len=3)
        if self.with_pos:
            output = seq + pos
        else:
            output = seq
        output = self.repeat_dim(output)
        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output)
            attentions.append(attention)
        return output[:, 1, :]



class SoftSelectAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SoftSelectAttention, self).__init__()

    def forward(self, support, query):
        """
        :param support: [few, dim]
        :param query: [batch, dim]
        :return:
        """
        query_ = query.unsqueeze(1).expand(query.size(0), support.size(0), query.size(1)).contiguous()  # [b, few, dim]
        support_ = support.unsqueeze(0).expand_as(query_).contiguous()  # [b, few, dim]

        scalar = support.size(1) ** -0.5  # dim ** -0.5
        score = torch.sum(query_ * support_, dim=2) * scalar
        att = torch.softmax(score, dim=1)

        center = torch.mm(att, support)
        return center


class SoftSelectPrototype(nn.Module):
    def __init__(self, r_dim):
        super(SoftSelectPrototype, self).__init__()
        self.Attention = SoftSelectAttention(hidden_size=r_dim)

    def forward(self, support, query):
        center = self.Attention(support, query)
        return center
