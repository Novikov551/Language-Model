import math

import torch


class RoPE(torch.nn.Module):

    #Параметры:
    #- theta: float — значение Θ для RoPE
    #- d_k: int — размерность векторов запросов и ключей (K_projection)
    #- max_seq_len: int — максимальная длина последовательности, которая будет подана на вход
    #- device: torch.device | None = None — устройство для хранения буфера
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, *args, **kwargs):
        # 1. Вызываем конструктор суперкласса
        super().__init__(*args, **kwargs)

        self.theta = theta

        assert d_k % 2 == 0, f"d_k must be even, got {d_k}"
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # нам нужно получить массив частот для каждой пары в эмбединге вектора запроса/ключа
        # она высчитывается для каждой пары по формуле 1 / (theta ** (2 * i/d_k), где ** возведение в степень,
        #  i - номер позиции пары, т.е допустим у нас есть вектор запроса [ 0, 1, 2, 3 ]
        #  у него есть 2 пары: x_0 = (0,1) с позицией 0 и x_1 = (2,3) с позицией 1
        # theta - частота (обычно 5000 - 20000)
        # d_k - длинна вектора запроса/ключа, в нашем случае 4, тк в векторе 4 элемента
        # Пример: частота для первой пары будет x_0_freq = 1.0/ (theta ** (2 * 0 / 4)) 
        # Для второй соответственно x_1_freq = 1.0/ (theta ** (2 * 1 / 4)) и.т.д
        # Для начала нужно сгенерировать массив с номерами позиций векторов запроса/ключа,
        # т.е если у нас вектор [ 0, 1, 2, 3 ], то нам нужен массив [ 0, 1 ] который делается так
        # torch.arange(0, d_k/2, 1), однако, учитывая что в формуле номер позиции умножается на 2 
        # можно сделать следующее torch.arange(0, d_k, 2), т.е мы получим следующий массив [0, 2]
        # и тогда из формулы можно убрать двойку когда мы возводим theta в степень и получить следующее
        # вместо (2 * (torch.arange(0, d_k/2, 1)) будет (torch.arange(0, d_k, 2)
        
        #генерируем массив степеней в которые будет возводиться theta
        pows = (torch.arange(0, d_k, 2, device=device).float() / d_k)

        #вычисляем частоты для позиций
        freq = 1.0 / (theta ** pows)

        #далее нам нужен массив с номерами всех возможных позиций слов, он ограничен размером контекста max_seq_len
        positions = torch.arange(max_seq_len, device=device)

        #затем нужно рассчитать углы на которые будут поворачиваться пары, угол рассчитывается по следующей формуле:
        #φ = p * ω_m, где ω_m - частота которую рассчитали ранее, p - позиция слова из positions
        angles = torch.outer(positions, freq)

        # Считаем значения синусов и косинусов (max_seq_len, d_k//2)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)

        #   Дублируем каждый угол для двух компонент пары
        #   .unsqueeze(-1) добавляет новую ось, т.е если у нас был след. тензор [ [1,2],[1,2]],
        # то теперь он будет таким [ [[1],[2]],[[1],[2]]
        #   .repeat(1, 1, 2) дублирует значение вдоль каждой оси указанное количество раз , т.е
        # если у нас был след. тензор [ [[1],[2]],[[1],[2]] то теперь он будет таким [ [[1,1],[2,2]],[[1,1],[2,2]]
        #   reshape собирает последние две оси (d_k//2, 2) в одну ось размера d_k//2 * 2 = d_k, т.е
        # если у нас был след. тензор  [ [[1,1],[2,2]],[[1,1],[2,2]] то теперь он будет таким [ [1,1,2,2],[1,1,2,2]]
        # 
        cos_vals = cos_vals.unsqueeze(-1).repeat(1, 1, 2).reshape((max_seq_len, d_k))  # (max_seq_len, d_k)
        sin_vals = sin_vals.unsqueeze(-1).repeat(1, 1, 2).reshape((max_seq_len, d_k))

        # 6. Регистрируем буферы (не обучаемые параметры)
        self.register_buffer("cos_buffer", cos_vals)
        self.register_buffer("sin_buffer", sin_vals)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Вспомогательная функция: для каждой пары (a, b) возвращает (-b, a).
        x: тензор произвольной формы, где последняя размерность равна d_k.
        Возвращает тензор той же формы.
        """
        # Разделяем на чётные и нечётные индексы
        x1 = x[..., 0::2]  # все чётные
        x2 = x[..., 1::2]  # все нечётные
        # Собираем обратно: (-x2, x1) вдоль последней оси
        rotated = torch.stack((-x2, x1), dim=-1).reshape(x.shape)
        return rotated
                 
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # cos и sin имеют форму (batch, seq_len, d_k) после индексации
        cos = self.cos_buffer[token_positions]  # (batch, seq_len, d_k)
        sin = self.sin_buffer[token_positions]

        # Если x имеет 4 измерения (batch, heads, seq_len, d_k), добавляем размерность heads
        if x.dim() == 4:
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, d_k)
            sin = sin.unsqueeze(1)  # (batch, 1, seq_len, d_k)

        # Применяем поворот: x_rot = x * cos + rotate_half(x) * sin
        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot