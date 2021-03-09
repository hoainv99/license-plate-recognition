from torchvision import models
import math
import torch
from torch import nn

class LanguageTransformer(nn.Module):
    def __init__(self,  vocab_size, 
                 d_model=256, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, 
                 dim_feedforward=2048, max_seq_length=256, 
                 pos_dropout=0.1, trans_dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embed_tgt = nn.Embedding(vocab_size, d_model)
        self._2D_pos_enc = _2DPositionalEncoding(d_model, pos_dropout, max_seq_length)
#        self.learned_pos_enc = LearnedPositionalEncoding(d_model, pos_dropout, max_seq_length)
        self.pos_enc = PositionalEncoding(d_model,pos_dropout,max_seq_length)
        self.transformer = nn.Transformer(d_model, nhead, 
                                          num_encoder_layers, num_decoder_layers, 
                                          dim_feedforward, trans_dropout)
        
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Shape:
            - src: (W, N, C)
            - tgt: (T, N) 
            - src_key_padding_mask: (N, S)
            - tgt_key_padding_mask: (N, T)
            - memory_key_padding_mask: (N, S)
            - output: (N, T, E)
            
        """

        tgt_mask = self.gen_nopeek_mask(tgt.shape[1]).to(src.device)



        src = self._2D_pos_enc(src*math.sqrt(self.d_model))
#        src = self.learned_pos_enc(src*math.sqrt(self.d_model))

        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model)).permute(1,0,2)
        output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
#        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)
        return self.fc(output)

    def gen_nopeek_mask(self, length):
        mask = (torch.triu(torch.ones(length, length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask
    
    def forward_encoder(self, src):
        src = self._2D_pos_enc(src*math.sqrt(self.d_model))
        memory = self.transformer.encoder(src)
        return memory
    
    def forward_decoder(self, tgt, memory):
        tgt_mask = self.gen_nopeek_mask(tgt.shape[0]).to(tgt.device)
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        
        output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask)
#        output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0, 1)

        return self.fc(output), memory
class _2DPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(_2DPositionalEncoding, self).__init__()
        #init positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        #init other
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(
            nn.Linear(d_model,d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2,d_model*2),
            nn.Sigmoid())
        self.d_model = d_model
    def forward(self,x):
        B,C,H,W = x.shape
        #position encoding
        h_encoding = self.pe[:H,:] #H,D
        w_encoding = self.pe[:W,:] #W,D
        h_encoding = h_encoding.unsqueeze(1) #H,1,D
        w_encoding = w_encoding.unsqueeze(0) #1,H,D
        h_encoding = h_encoding.unsqueeze(0).repeat(B,1,1,1) #B,H,1,D
        w_encoding = w_encoding.unsqueeze(1).repeat(B,1,1,1) #B,1,W,D
        
        #adaptive position encoding
        
        inter = self.avg(x).view(B,-1) #B,Hidden
        alpha = self.dense(inter) #B,d_model*2
        alpha = alpha.reshape(-1,2,1,self.d_model)#B,2,1,d_model
        x = x.permute(0,2,3,1)
        x = x + alpha[:,0:1,:,:]*h_encoding+ alpha[:,1:2,:,:]*w_encoding 
        return x.view(-1,H*W,self.d_model).permute(1,0,2)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1,max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
                           
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_len, d_model)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, d_model, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta  