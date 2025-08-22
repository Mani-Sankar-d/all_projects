import torch 
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        self.W_q = nn.Linear(emb_dim, emb_dim)
        self.W_k = nn.Linear(emb_dim, emb_dim)
        self.W_v = nn.Linear(emb_dim, emb_dim)
        self.W_o = nn.Linear(emb_dim,emb_dim)
        self.ff1 = nn.Linear(emb_dim, 4*emb_dim)
        self.ff2 = nn.Linear(4*emb_dim,emb_dim)
        self.n_heads = n_heads
        self.head_dim = (emb_dim//n_heads)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.gelu = nn.GELU()
    def forward(self, token_embeddings):
        batch_size, max_len, emb_dim = token_embeddings.shape
        Q = self.W_q(token_embeddings)
        K = self.W_k(token_embeddings)
        V = self.W_v(token_embeddings)

        Q = Q.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)
        K = K.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)
        V = V.view(-1,max_len,self.n_heads, self.head_dim).transpose(1,2)

        attn_score = torch.softmax(torch.matmul(Q,K.transpose(-2,-1))/self.head_dim**0.5,dim=-1)
        new_embeddings = attn_score@V
        new_embeddings = new_embeddings.transpose(1,2).view(-1,max_len,emb_dim)
        new_embeddings = self.W_o(new_embeddings)
        new_embeddings = self.norm1(token_embeddings + new_embeddings)
        old_embeddings = new_embeddings
        new_embeddings = self.gelu(self.ff1(new_embeddings))
        new_embeddings = self.ff2(new_embeddings)
        new_embeddings = self.norm2(new_embeddings+old_embeddings)

        return new_embeddings


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(max_len, embed_dim)
        self.seg_embeddings = nn.Embedding(2, embed_dim)
        self.dense_layer = nn.Linear(embed_dim,vocab_size)

        self.encoder1 = Encoder(embed_dim,n_heads)
        self.encoder2 = Encoder(embed_dim,n_heads)
        self.encoder3 = Encoder(embed_dim,n_heads)
        self.encoder4 = Encoder(embed_dim,n_heads)
        
        self.encoder5 = Encoder(embed_dim,n_heads)
        self.encoder6 = Encoder(embed_dim,n_heads)
        self.encoder7 = Encoder(embed_dim,n_heads)
        self.encoder8 = Encoder(embed_dim,n_heads)

        
        self.encoder9 = Encoder(embed_dim,n_heads)
        self.encoder10 = Encoder(embed_dim,n_heads)
        self.encoder11 = Encoder(embed_dim,n_heads)
        self.encoder12 = Encoder(embed_dim,n_heads)

        self.encoder_stack = nn.Sequential(
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder5,
            self.encoder6,
            self.encoder7,
            self.encoder8,
            self.encoder9,
            self.encoder10,
            self.encoder11,
            self.encoder12
        )

    def forward(self, token_ids, segment_ids, mask_ids):
        segment_embeddings = self.seg_embeddings(segment_ids)
        batch_size ,max_len = token_ids.shape
        token_embeddings = self.embeddings(token_ids)
        pos_embeddings = self.pos_embeddings(torch.arange(0,max_len,device=token_ids.device).unsqueeze(0).expand(batch_size,-1))
        final_embeddings = token_embeddings+segment_embeddings+pos_embeddings
        final_embeddings = self.encoder_stack(final_embeddings)
        mask_ids = mask_ids.unsqueeze(-1).expand(-1,-1,self.embed_dim)
        out = torch.gather(final_embeddings,1,mask_ids)
        res = self.dense_layer(out)
        return res


