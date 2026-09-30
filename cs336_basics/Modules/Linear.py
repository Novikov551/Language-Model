import math

import torch

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None, *args, **kwargs):

        # 1. Вызываем конструктор суперкласса
        super().__init__(*args, **kwargs)

        # 2. Создаём тензор для весов нужной формы и типа
        weights = torch.empty(out_features, in_features, device=device, dtype=dtype)

        std = math.sqrt(2.0 / (in_features + out_features))

        # 3. Инициализируем веса с помощью truncated normal
        torch.nn.init.trunc_normal_(weights, mean=0.0, std=std, a=-3*std, b=3*std)

        # 4. Оборачиваем в nn.Parameter и сохраняем как self.W
        self.W = torch.nn.Parameter(weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T