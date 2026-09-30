import math
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F

def soft_max(x: torch.Tensor, dim: int) -> torch.Tensor:

    max_val = torch.amax(x, dim=dim, keepdim=True)

    stable = x - max_val

    exp = torch.exp(stable)
    exp_sum = torch.sum(exp, dim=dim, keepdim=True)

    return exp / exp_sum

def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:

    logits = Q @ K.transpose(-2,-1)

    d_k = Q.shape[-1]

    normalized = logits / math.sqrt(d_k)

    if mask is not None:
        normalized = normalized.masked_fill(~mask, -torch.inf)

    attention = soft_max(normalized, -1)

    return attention @ V

#log_softmax(o) j=oj−max(o)−log((k=1 ∑ V) exp(ok−max(o))).
def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет кросс-энтропию между логитами и целевыми индексами.
    
    Аргументы:
        logits: тензор формы (..., vocab_size)
        targets: тензор формы (...) - целочисленные индексы правильных классов
    
    Возвращает:
        Скаляр - среднее значение потерь по всем элементам пакета.
    """
    # 1. Вычитаем максимум для численной стабильности (по последней оси)

    # Ищем максимум среди всех логитов
    logits_max = logits.amax(dim=-1, keepdim=True)

    # Вычитаем максимум из всех элементов
    logits_stable = logits - logits_max   # форма (..., vocab_size)
    
    # 2. Вычисляем сумму экспонент S=(k=1 ∑ V)exp⁡((~o)k)S=(k=1 ∑ V) exp((~o)k).
    # Вычисляем log-sum-exp: log( sum(exp(logits_stable)) )
    exp_logits = torch.exp(logits_stable)
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)   # (..., 1)
    log_sum_exp = torch.log(sum_exp)                 # (..., 1)
    
    # 3. Логарифм softmax: log_softmax = logits_stable - log_sum_exp
    log_softmax = logits_stable - log_sum_exp        # (..., vocab_size)
    
    # 4. Берём значения для правильных индексов
    #    targets нужно привести к форме (..., 1) для gather
    targets_unsqueezed = targets.unsqueeze(-1)       # (..., 1)
    log_probs = log_softmax.gather(dim=-1, index=targets_unsqueezed)  # (..., 1)
    log_probs = log_probs.squeeze(-1)                # убираем лишнюю размерность
    
    # 5. Потери = -log_prob, затем усредняем
    loss = -log_probs
    return loss.mean()

def perplexity(cross_entropy_loss: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет перплексию как exp(средняя кросс-энтропия).
    
    Args:
        logits: тензор формы (..., vocab_size)
        targets: тензор формы (...)
    Returns:
        скалярная перплексия
    """
    
    return torch.exp(cross_entropy_loss)

def lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    warmup_steps: int,
    cosine_steps: int,
) -> float:
    """
    Возвращает скорость обучения на шаге t по расписанию с прогревом и косинусным отжигом.
    """
    if t < warmup_steps:
        # Прогрев: линейно от 0 до alpha_max
        return (t / warmup_steps) * alpha_max
    elif t <= cosine_steps:
        # Косинусный отжиг: от alpha_max до alpha_min
        progress = (t - warmup_steps) / (cosine_steps - warmup_steps)
        cosine = math.cos(math.pi * progress)
        return alpha_min + 0.5 * (1 + cosine) * (alpha_max - alpha_min)
    else:
        # После отжига — постоянно alpha_min
        return alpha_min
    

def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6) -> None:
    """
    Клиппирует градиенты всех параметров по L2-норме.
    parameters: итерируемый объект параметров (например, model.parameters())
    max_norm: максимально допустимая норма
    eps: для численной стабильности
    """
    total_norm_sq = 0.0
    # Собираем градиенты в список (только не-None)
    grads = []
    for p in parameters:
        if p.grad is not None:
            grad = p.grad.data
            grads.append(grad)
            total_norm_sq += grad.pow(2).sum().item()
    norm = (total_norm_sq ** 0.5)
    if norm > max_norm:
        coeff = max_norm / (norm + eps)
        for grad in grads:
            grad.mul_(coeff)


def get_batch(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length - 1
    if max_start < 0:
        raise ValueError("Dataset too short for given context_length")
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    inputs_list = []
    targets_list = []
    for start in starts:
        inputs_list.append(dataset[start:start+context_length])
        targets_list.append(dataset[start+1:start+context_length+1])
    inputs_np = np.stack(inputs_list)   # (batch, context_length)
    targets_np = np.stack(targets_list)
    inputs_t = torch.tensor(inputs_np, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets_np, dtype=torch.long, device=device)
    return inputs_t, targets_t

def save_checkpoint(model, optimizer, iteration, out):
    """
    Сохраняет контрольную точку в файл out.
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    iteration: int
    out: str или файловый объект
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    """
    Загружает контрольную точку из src и восстанавливает модель и оптимизатор.
    Возвращает iteration.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

@torch.no_grad()
def evaluate_loss(model, val_data, batch_size, context_length, device, num_batches=5):
    """
    Вычисляет средний loss на валидационном наборе (несколько батчей).
    """
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        inputs, targets = get_batch(val_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches

def top_p_filtering(logits, top_p=0.9, filter_value=-float('Inf')):
    """
    Фильтрует логиты, оставляя только маловероятные токены за порогом top-p.
    
    Args:
        logits: тензор формы (vocab_size,)
        top_p: порог суммы вероятностей (0.0..1.0)
        filter_value: значение, которым заменяются отфильтрованные логиты (обычно -inf)
    Returns:
        отфильтрованные логиты (той же формы)
    """
    if top_p >= 1.0:
        return logits
    # Сортируем вероятности по убыванию
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # Кумулятивная сумма
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    # Убираем токены, после которых cumsum > top_p
    mask = cumsum_probs > top_p
    # Сдвигаем маску вправо, чтобы оставить первый токен, превышающий порог
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    # Применяем маску к исходным логитам (через индексы)
    filtered_logits = logits.clone()
    filtered_logits[sorted_indices[mask]] = filter_value
    return filtered_logits






@torch.no_grad()
def generate(
    model,
    prompt_ids,           # torch.LongTensor формы (1, seq_len) или (seq_len,)
    max_new_tokens=100,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    stop_token_id=None,
    device=None
):
    """
    Генерирует продолжение последовательности токенов.
    
    Args:
        model: обученная языковая модель (TransformerLanguageModel)
        prompt_ids: тензор с начальными токенами (размерность может быть 1D или 2D с batch=1)
        max_new_tokens: максимальное количество генерируемых токенов
        temperature: >0, чем меньше, тем более детерминировано
        top_p: nucleus sampling (0..1), 1.0 означает отключено
        top_k: если задано, оставляет только top_k наиболее вероятных токенов
        stop_token_id: если не None, генерация останавливается при появлении этого токена
        device: устройство (если None, берётся из параметров модели)
    Returns:
        torch.LongTensor формы (1, total_length) – промпт + сгенерированные токены
    """
    if device is None:
        device = next(model.parameters()).device
    # Приводим к виду (1, seq_len)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    prompt_ids = prompt_ids.to(device)
    generated = prompt_ids
    model.eval()

    for _ in range(max_new_tokens):
        # Обрезаем до контекстной длины, если модель имеет ограничение
        # (обычно в модели есть `context_length`, можно использовать атрибут)
        if hasattr(model, 'context_length'):
            context_len = model.context_length
            if generated.shape[1] > context_len:
                context = generated[:, -context_len:]
            else:
                context = generated
        else:
            context = generated

        # Прямой проход
        logits = model(context)                 # (1, seq_len, vocab_size)
        next_logits = logits[0, -1, :]          # (vocab_size,)
        # Температура
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Вырожденный случай: greedy
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            break
        # Top‑k (если задан)
        if top_k is not None and top_k > 0:
            topk_vals, topk_idx = torch.topk(next_logits, top_k)
            mask = torch.ones_like(next_logits, dtype=torch.bool)
            mask[topk_idx] = False
            next_logits[mask] = -float('inf')
        # Top‑p (nucleus)
        if top_p is not None and top_p < 1.0:
            next_logits = top_p_filtering(next_logits, top_p)
        # Преобразуем в вероятности и семплируем
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)   # (1,)
        # Добавляем в последовательность
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        # Остановка, если встретили стоп-токен
        if stop_token_id is not None and next_token.item() == stop_token_id:
            break
    return generated