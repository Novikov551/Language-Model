import math
from typing import Optional

import torch

from cs336_basics.Modules.Functions import scaled_dot_product_attention
from cs336_basics.Modules.Linear import Linear
from cs336_basics.Modules.RoPE import RoPE

class MultiheadSelfAttention(torch.nn.Module):

    def __init__(self, 
                  d_model: int, 
                  num_heads: int, 
                  mask: bool, 
                  max_seq_len: int = 2048, 
                  theta: Optional[float] = None,
                  device = None,
                  dtype = None, 
                  *args, 
                  **kwargs):
        super().__init__(*args, **kwargs)
        self.W_Q =  Linear(d_model, d_model, device, dtype)
        self.W_K =  Linear(d_model, d_model, device, dtype)
        self.W_V =  Linear(d_model, d_model, device, dtype)
        self.W_O =  Linear(d_model, d_model, device, dtype)
        self.register_buffer("causal_mask", torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)))

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads

        if theta is not None:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)
        else:
            self.rope = None

        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_model = d_model
        self.mask = mask
        self.device = device
        self.theta = theta
        
    def forward(self, x: torch.Tensor,
                token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q_heads = split_heads(Q, self.num_heads)
        K_heads = split_heads(K, self.num_heads)
        V_heads = split_heads(V, self.num_heads)

        if self.rope is not None and token_positions is not None:
            Q_heads = self.rope(Q_heads, token_positions)
            K_heads = self.rope(K_heads, token_positions)
        
        mask = None
        if(self.mask):
            seq_len = x.shape[-2]
            mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
            
        heads = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)

        heads = combine_heads(heads, self.d_model)
        return self.W_O(heads)
    
def combine_heads(x, d_model):
    """
    Объединяет головы многоголового внимания обратно в d_model.
    
    Аргументы:
        x: тензор формы (..., num_heads, seq_len, head_dim)
        d_model: конечная размерность (num_heads * head_dim)
    
    Возвращает:
        тензор формы (..., seq_len, d_model)
    """
    # Меняем оси: (..., num_heads, seq_len, head_dim) -> (..., seq_len, num_heads, head_dim)
    x = x.transpose(-2, -3)
    # Объединяем последние две оси: (..., seq_len, d_model)
    x = x.contiguous().view(*x.shape[:-2], d_model)
    return x

def split_heads(x, num_heads):

    # x.shape = (..., seq_len, d_model)

    *leading_dims, seq_len, d_model = x.shape

    head_dim = d_model // num_heads

    # new_shape = (*leading_dims, seq_len, num_heads, head_dim)

    x = x.view(*leading_dims, seq_len, num_heads, head_dim)

    # Поменять местами оси: (..., seq_len, num_heads, head_dim) -> (..., num_heads, seq_len, head_dim)
    x = x.transpose(-2, -3)  # меняем предпоследнюю и пред-предпоследнюю оси

    return x