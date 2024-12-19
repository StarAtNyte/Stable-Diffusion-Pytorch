import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # Combines Wq, Wk, Wv matrices into a single matrix for efficient computation
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Output projection (Wo matrix)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        # Input shape: (batch_size, seq_len, d_embed)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        # Shape for multi-head computation
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # Split input into query, key, value projections
        # Shape: (batch_size, seq_len, d_embed) -> (batch_size, seq_len, 3 * d_embed) -> 3 x (batch_size, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # Reshape for multi-head attention and transpose to get heads as batch dim
        # Shape: (batch_size, n_heads, seq_len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention scores
        # Shape: (batch_size, n_heads, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Create causal mask to prevent attending to future tokens
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        # Scale attention scores by dimension of key vectors
        weight /= math.sqrt(self.d_head)

        # Apply softmax to get attention weights
        weight = F.softmax(weight, dim=-1)

        # Apply attention weights to values
        # Shape: (batch_size, n_heads, seq_len, d_head)
        output = weight @ v

        # Reshape and transpose back to original dimensions
        # Shape: (batch_size, seq_len, d_embed)
        output = output.transpose(1, 2).reshape(input_shape)

        # Final output projection
        output = self.out_proj(output)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, y):
        # x: Query tensor (batch_size, seq_len_q, d_embed)
        # y: Key/Value tensor (batch_size, seq_len_kv, d_cross)
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # Project inputs to query, key, and value
        # Shape: (batch_size, seq_len_q/kv, d_embed)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape for multi-head attention
        # Shape: (batch_size, n_heads, seq_len_q/kv, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        # Compute scaled dot-product attention
        # Shape: (batch_size, n_heads, seq_len_q, seq_len_kv)
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        # Apply attention weights to values
        # Shape: (batch_size, n_heads, seq_len_q, d_head)
        output = weight @ v
        
        # Reshape back to original dimensions
        # Shape: (batch_size, seq_len_q, d_embed)
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        return output