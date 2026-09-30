import torch

class Embedding(torch.nn.Module):

    # num_embeddings - по сути размер словаря
    # embedding_dim - размер вектора
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None, *args, **kwargs):
        # 1. Вызываем конструктор суперкласса
        super().__init__(*args, **kwargs)

        # 2. Создаём тензор для весов нужной формы и типа
        weights = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)

        # 3. Инициализируем веса с помощью truncated normal
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

        # 4. Оборачиваем в nn.Parameter и сохраняем как self.W
        self.W = torch.nn.Parameter(weights)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.W[token_ids]