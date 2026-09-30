import math
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

class AdamWOptimizer(Optimizer):
    """
    AdamW оптимизатор.
    Параметры:
    - params: итерируемый объект параметров (например, model.parameters())
    - lr: скорость обучения (alpha)
    - betas: кортеж (beta1, beta2), обычно (0.9, 0.999) или (0.9, 0.95)
    - eps: для численной стабильности (1e-8)
    - weight_decay: коэффициент распада весов (лямбда)
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Недопустимая скорость обучения: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Недопустимый beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Недопустимый beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Недопустимый eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Недопустимый weight_decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Выполняет один шаг обновления параметров.
        closure: для повторного вычисления лосса (нам не нужна)
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW не поддерживает разреженные градиенты")

                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                t = state["step"] + 1   # текущий шаг (начинаем с 1)

                # Обновление моментов
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Коррекция смещения
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # Обновление параметров 
                p.data.addcdiv_(m_hat, (v_hat.sqrt() + eps), value=-lr)

                # Применяем weight decay отдельно 
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                state["step"] = t

        return loss