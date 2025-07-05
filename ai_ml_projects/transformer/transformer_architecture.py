import torch
import torch.nn as nn
import math
def get_pos_enc(x):
    batch_size, seq_len, embed_size = x.shape
    pos_enc = []
    for pos in range(seq_len):
        row = []
        for i in range(0,embed_size):
            if(i%2==0): row.append(math.sin(pos/10000**(i/embed_size))) 
            else: row.append(math.cos(pos/10000**(i/embed_size)))
        pos_enc.append(row)
    pos_enc = torch.tensor(pos_enc, dtype=x.dtype, device=x.device)
    pos_enc = pos_enc.unsqueeze(0)
    return x+pos_enc


class encoder(nn.Module):
    def __init__(self, heads, embed_size, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.head_size=embed_size//heads
        self.heads=heads
        self.embed_size=embed_size
        self.q_layer = nn.Linear(embed_size,embed_size)
        self.k_layer = nn.Linear(embed_size, embed_size)    
        self.v_layer = nn.Linear(embed_size, embed_size)
        self.dropout=nn.Dropout(dropout)
        self.fc_out=nn.Linear(embed_size,embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_size)
        )
    def forward(self,x):
        batch_size, seq_len,_=x.shape
        Q=self.q_layer(x)
        K=self.k_layer(x)
        V=self.v_layer(x)

        Q=Q.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)
        K=K.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)
        V=V.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1))/(self.head_size**0.5)
        attention = torch.softmax(scores,dim=-1)
        out=torch.matmul(attention, V)
        out=out.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_size)
        out =self.dropout(self.fc_out(out))
        x=self.norm1(x+out)
        out=self.dropout(self.ff(x))
        final_output = self.norm2(x+out)
        return final_output
    



class encoderstack(nn.Module):
    def __init__(self, nx, heads, embed_size, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.layers  = nn.ModuleList([
            encoder(heads, embed_size, ff_dim,dropout)
            for _ in range(nx)
        ])
    def forward(self, x):
        x=get_pos_enc(x)
        for layer in self.layers:
            x=layer(x)
        return x

class decoder(nn.Module):
    def __init__(self, heads, embed_size, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.head_size=embed_size//heads
        self.heads=heads
        self.embed_size=embed_size

        #self attn
        self.q_layer = nn.Linear(embed_size,embed_size)
        self.k_layer = nn.Linear(embed_size, embed_size)    
        self.v_layer = nn.Linear(embed_size, embed_size)

        #cross attn
        self.cross_q = nn.Linear(embed_size, embed_size)
        self.cross_k = nn.Linear(embed_size, embed_size)    
        self.cross_v = nn.Linear(embed_size, embed_size)

        #universal
        self.fc_out=nn.Linear(embed_size,embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,embed_size)
        )
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,enc_out):
        batch_size, seq_len,_=x.shape
        x=get_pos_enc(x)
        #self attn
        Q=self.q_layer(x)
        K=self.k_layer(x)
        V=self.v_layer(x)
        Q=Q.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)
        K=K.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)
        V=V.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1))/(self.head_size**0.5)
        destroyer = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1).to(device)
        mask_scores = scores+destroyer
        attention = torch.softmax(mask_scores,dim=-1)
        out=torch.matmul(attention, V)
        out=out.transpose(1,2).contiguous().view(batch_size,seq_len,self.embed_size)
        out =self.dropout(self.fc_out(out))
        x=self.norm1(x+out)

        #cross attn
        enc_seq_len = enc_out.shape[1]
        Q=self.cross_q(x)
        K=self.cross_k(enc_out)
        V=self.cross_v(enc_out)
        Q=Q.view(batch_size, seq_len, self.heads, self.head_size).transpose(1,2)
        K=K.view(batch_size, enc_seq_len, self.heads, self.head_size).transpose(1,2)
        V=V.view(batch_size, enc_seq_len, self.heads, self.head_size).transpose(1,2)
        scores=torch.matmul(Q,K.transpose(-2,-1))/(self.head_size**0.5)
        attention=torch.softmax(scores, dim=-1)
        out=torch.matmul(attention, V)
        out=self.dropout(self.fc_out(out))
        x=self.norm2(x+out)


        out=self.ff(x)
        x=self.norm3(x+out)
        return x


class decoderstack(nn.Module):
    def __init__(self,nx,heads, embed_size, ff_dim=2048, dropout=0.1,vocab_size):
        super().__init__()
        self.layers = nn.ModuleList([
            decoder(heads,embed_size, ff_dim,dropout)
            for _ in range(nx)
        ])
        self.final = nn.Linear(embed_size, vocab_size) 
    def forward(self, x,enc_out):
        for layer in self.layers:
            x=layer(x,enc_out)
        x=torch.softmax(self.final(x), dim=-1)
        return x