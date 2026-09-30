import torch

from cs336_basics.Modules.Embedding import Embedding
from cs336_basics.Modules.Linear import Linear
from cs336_basics.Modules.RMSNorm import RMSNorm
from cs336_basics.Modules.TransformerBlock import TransformerBlock

class TransformerLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        context_length: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        theta: float = 10000,      
        eps: float = 1e-5,
        mask: bool = True,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embeddings = Embedding(vocab_size, d_model, device=device, dtype = dtype)
        self.out_norm = RMSNorm(d_model, eps, device=device, dtype = dtype)
        self.out_linear = Linear(d_model, vocab_size, device=device, dtype = dtype)
        self.mask = mask
        self.transformer_blocks = torch.nn.ModuleList([])
        
        for _ in range(num_layers):
            self.transformer_blocks.append(TransformerBlock(d_model, num_heads, d_ff, context_length, True, theta, eps,device, dtype = dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = self.embeddings(x)
        for block in self.transformer_blocks:
            token_positions = None
            if(self.mask):
                *leading_dims, seq_len = x.shape
                token_positions = torch.arange(seq_len, device=x.device).expand(*leading_dims, seq_len)
            input = block(input, token_positions)
        out_norm = self.out_norm(input)
        return self.out_linear(out_norm)