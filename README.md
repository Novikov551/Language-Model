## Стек

- Python 3.12+, PyTorch 2.11
- uv (пакетный менеджер)
- numpy, regex, einops, tqdm, wandb, matplotlib, flask

## Структура проекта

```
cs336_basics/
├── Modules/
│   ├── BPETokenizer.py        — BPE-токенизатор (обучение + encode/decode)
│   ├── ChunkIdentifier.py     — параллельная нарезка файлов на чанки для претокенизации
│   ├── Linear.py              — линейный слой (truncated normal init)
│   ├── Embedding.py           — embedding-слой
│   ├── RMSNorm.py             — Root Mean Square Layer Normalization
│   ├── RoPE.py                — Rotary Position Embeddings
│   ├── MultiheadSelfAttention.py — multi-head self-attention с каузальной маской
│   ├── SwiGLU.py              — SwiGLU feed-forward блок
│   ├── TransformerBlock.py    — Transformer-блок (pre-norm, residual connections)
│   ├── TransformerLM.py       — полная языковая модель (embedding → N blocks → norm → linear)
│   ├── AdamWOptimizer.py      — реализация AdamW с нуля
│   └── Functions.py           — softmax, cross-entropy, cosine LR schedule, gradient clipping, checkpointing, генерация текста
├── ModelTraining.py           — CLI для обучения (argparse, wandb, чекпоинты, метрики в CSV)
├── BPETokenizerTraining.py    — CLI для обучения BPE-токенизатора
├── TextEncoding.py            — скрипт кодирования текста в токены (.txt → .npy)
├── TextGenerate.py            — интерактивная генерация текста из обученной модели
├── PlotMetrics.py             — визуализация loss/perplexity из CSV
├── app.py                     — Flask-приложение с веб-интерфейсом для генерации
├── Checkpoints/               — сохранённые чекпоинты модели
├── Datasets/                  — TinyStories (train/valid, .txt и .npy)
└── Metrics/                   — CSV с метриками обучения
```

## Архитектура модели

Transformer LM с pre-norm конфигурацией:

1. **Embedding** — токены → векторы (vocab_size → d_model)
2. **N Transformer-блоков**, каждый:
   - RMSNorm → Multi-Head Self-Attention (с RoPE) → Residual
   - RMSNorm → SwiGLU FFN → Residual
3. **RMSNorm** → **Linear** (d_model → vocab_size) — logits

Каузальная маска в attention обеспечивает автогрессивное предсказание.

## Гиперпараметры (TinyStories)

| Параметр       | Значение |
|----------------|----------|
| vocab_size     | 10 000   |
| d_model        | 512      |
| num_heads      | 16       |
| num_layers     | 4        |
| d_ff           | 1 344    |
| context_length | 256      |
| theta (RoPE)   | 10 000   |

## Быстрый старт

```bash
# Установка зависимостей
uv sync

# Обучение BPE-токенизатора
uv run python -m cs336_basics.BPETokenizerTraining --vocab 10000

# Кодирование данных
uv run python -m cs336_basics.TextEncoding

# Обучение модели
uv run python -m cs336_basics.ModelTraining \
    --train_data cs336_basics/Datasets/TinyStories-train.npy \
    --val_data cs336_basics/Datasets/TinyStories-valid.npy \
    --batch_size 32 \
    --context_length 256 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 4 \
    --d_ff 1344 \
    --max_steps 5000

# Генерация текста
uv run python -m cs336_basics.TextGenerate

# Визуализация метрик
uv run python -m cs336_basics.PlotMetrics
```

## Запуск тестов

```bash
uv run pytest
```

## Веб-интерфейс

```bash
uv run python -m cs336_basics.app
```

Flask-приложение с UI для генерации текста по промпту.

## Реализованные компоненты

- [x] BPE-токенизатор (обучение, encode, decode, претокенизация с regex, специальные токены, параллелизация через ProcessPoolExecutor)
- [x] Softmax, scaled dot-product attention, cross-entropy loss
- [x] Linear, Embedding (с нуля, без nn.Linear/nn.Embedding)
- [x] RMSNorm, RoPE, SwiGLU
- [x] Multi-Head Self-Attention с каузальной маской
- [x] Transformer LM (pre-norm)
- [x] AdamW оптимизатор (с нуля)
- [x] Cosine LR schedule, gradient clipping
- [x] Training loop с чекпоинтами
- [x] Text generation (greedy/sampling)
- [x] Метрики (loss, perplexity, wandb-логирование)