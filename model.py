import torch
import torch.nn as nn
import math

##
## Simple Model
## 

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, max_seq_len=1024):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            GPT2Block(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(pos_ids)
        
        x = token_emb + pos_emb
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(att)

class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x