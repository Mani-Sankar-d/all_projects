import torch
import torch.nn as nn
from data import input_ids_dataset, labels_dataset, pad_id
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, TensorDataset
class Decoder(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim//n_heads
        self.Q =  nn.Linear(emb_dim, emb_dim)
        self.K =  nn.Linear(emb_dim, emb_dim)
        self.V =  nn.Linear(emb_dim, emb_dim)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(emb_dim, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)

        self.ff = nn.Sequential(
            nn.Linear(emb_dim,4*emb_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(emb_dim*4,emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)
    def _build_causal_mask(self, seq_len, device):
        mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self,embeddings):
        batch_size, seq_len, emb_dim = embeddings.shape
        n_heads = self.n_heads
        head_dim = self.head_dim
        old_emb = embeddings
        q = self.Q(embeddings)
        k = self.K(embeddings)
        v = self.V(embeddings)
        
        q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, n_heads, head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, n_heads, head_dim).transpose(1,2)

        attn_score = torch.matmul(q,torch.transpose(k,-1,-2))/(head_dim**0.5)
        attn_score = attn_score + self._build_causal_mask(seq_len,embeddings.device)
        attn_score = torch.softmax(attn_score,dim=3)
        new_emb = torch.matmul(attn_score,v)
        new_emb = new_emb.transpose(1,2).contiguous().view(batch_size,seq_len,n_heads*head_dim)
        new_emb = self.drop(self.fc(new_emb))
        new_emb += old_emb
        old_emb = self.norm1(new_emb)
        new_emb = self.ff(old_emb)
        new_emb+=old_emb
        new_emb = self.norm2(new_emb)
        return new_emb
        


class Model(nn.Module):
    def __init__(self, vocab_size,max_len,emb_dim, n_heads):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, emb_dim)
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.decoder1 = Decoder(emb_dim, n_heads)
        self.decoder2 = Decoder(emb_dim, n_heads)
        self.decoder3 = Decoder(emb_dim, n_heads)
        self.decoder4 = Decoder(emb_dim, n_heads)
        self.decoder5 = Decoder(emb_dim, n_heads)
        self.decoder6 = Decoder(emb_dim, n_heads)
        self.final = nn.Linear(emb_dim, vocab_size)
    def forward(self,ids):
        batch_size, seq_len = ids.shape
        pos_emb = torch.arange(0,seq_len, device=ids.device)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, seq_len)
        tok_emd = self.token_emb(ids) + self.pos_emb(pos_emb) 

        emb = self.decoder1(tok_emd)
        emb = self.decoder2(emb)
        emb = self.decoder3(emb)
        emb = self.decoder4(emb)
        emb = self.decoder5(emb)
        emb = self.decoder6(emb)
        out = self.final(emb)
        return out