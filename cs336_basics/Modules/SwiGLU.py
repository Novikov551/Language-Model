import torch

from cs336_basics.Modules.Linear import Linear

class SwiGLU(torch.nn.Module):

    #Параметры:
    #- d_model: int — скрытая размерность модели
    #- device: torch.device | None = None — устройство для хранения параметров
    #- dtype: torch.dtype | None = None — тип данных параметров
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None, *args, **kwargs):
        # 1. Вызываем конструктор суперкласса
        super().__init__(*args, **kwargs)

        # Нужно установить d_ff примерно равным (8/3) * d_model и сделать его кратным 64
        if d_ff is None:
            d_ff = ((int((8 * d_model) / 3) + 63) // 64) * 64

        self.W1 = Linear(d_model, d_ff, device, dtype)   # in=d_model, out=d_ff
        self.W3 = Linear(d_model, d_ff, device, dtype)
        self.W2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        #пропускаем входной тензор через линейный слой 1 чтобы увеличить его до размера прмиерно 8/3 * d_model (кратного 64)
        w1_raw = self.W1(x)

        #получаем сигмоиду для результата
        w1_sigmoid = w1_raw * torch.sigmoid(w1_raw)

        #пропускаем входной тензор через линейный слой 3
        w3_raw = self.W3(x)

        #делаем поэлеметное умножение для результата
        w13_raw = w1_sigmoid * w3_raw

        #пропускаем результат через линейный слой 2 чтобы вернуть ему форму входного тензора
        return self.W2(w13_raw)