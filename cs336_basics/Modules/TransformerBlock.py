import torch

from cs336_basics.Modules.MultiheadSelfAttention import MultiheadSelfAttention
from cs336_basics.Modules.RMSNorm import RMSNorm
from cs336_basics.Modules.SwiGLU import SwiGLU

class TransformerBlock(torch.nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 max_seq_len: int = 2048,   
                 mask: bool = True,        
                 theta: float = None,      
                 eps: float = 1e-5,
                 device=None, 
                 dtype=None,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_attention_normalization = RMSNorm(d_model, eps, device, dtype)
        self.multi_head_self_attention = MultiheadSelfAttention(d_model, num_heads, mask, max_seq_len, theta, device, dtype)
        self.pre_swiglu_normalization =  RMSNorm(d_model, eps, device, dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device, dtype)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:

        pre_attention_normalization = self.pre_attention_normalization(x)
        multi_head_self_attention = self.multi_head_self_attention(pre_attention_normalization, token_positions)
        x = x + multi_head_self_attention

        pre_swiglu_normalization = self.pre_swiglu_normalization(x)
        swiglu = self.swiglu(pre_swiglu_normalization)
        x = x + swiglu

        return x