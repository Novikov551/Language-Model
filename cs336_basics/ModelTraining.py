import os
import sys
import math
import argparse
import csv
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.Modules.AdamWOptimizer import AdamWOptimizer
from cs336_basics.Modules.Functions import (
    cross_entropy_loss,
    evaluate_loss,
    get_batch,
    gradient_clipping,
    load_checkpoint,
    lr_cosine_schedule,
    save_checkpoint,
)
from cs336_basics.Modules.TransformerLM import TransformerLanguageModel

# Опционально: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def set_seed(seed: int):
    """Фиксирует все случайные seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")

    # --- Данные ---
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (.bin or .npy)")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length (sequence length)")

    # --- Модель ---
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer blocks")
    parser.add_argument("--d_ff", type=int, default=None, help="FFN inner dimension (auto if None)")
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta parameter")
    parser.add_argument("--no_rope", default= False, action="store_true", help="Disable RoPE (sets theta=None)")

    # --- Оптимизатор ---
    parser.add_argument("--lr", type=float, default=3e-4, help="Max learning rate")
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Min learning rate after cosine")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95), help="AdamW betas")
    parser.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")

    # --- Обучение ---
    parser.add_argument("--max_iters", type=int, default=10000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate val loss every N steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log train loss every N steps")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device: cpu, cuda, mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto if None)")

    args = parser.parse_args()

    # Фиксация seed
    set_seed(args.seed)

    # Создаём директорию для чекпоинтов и CSV
    os.makedirs("./Metrics", exist_ok=True)

    # CSV файл для метрик
    csv_path = os.path.join("./Metrics", "metrics.csv")
    # Если файл не существует, создаём с заголовками
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss", "train_ppl", "val_ppl", "lr", "time_seconds"])

    # Weights & Biases
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb not installed. Install with `pip install wandb` to enable logging.", file=sys.stderr)

    # Загрузка данных через memmap
    print("Loading data...")
    train_data = np.load(args.train_data, mmap_mode='r') if args.train_data.endswith('.npy') else np.memmap(args.train_data, dtype=np.uint16, mode='r')
    val_data = np.load(args.val_data, mmap_mode='r') if args.val_data.endswith('.npy') else np.memmap(args.val_data, dtype=np.uint16, mode='r')
    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Создание модели
    model = TransformerLanguageModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        context_length=args.context_length,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        theta=None if args.no_rope else args.rope_theta,
        eps=1e-5,
        mask=True,
        device=args.device,
        dtype=torch.float32,
    )
    model.to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params:,} parameters")

    # Оптимизатор
    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # Функция расписания
    def scheduler(step):
        return lr_cosine_schedule(
            step,
            alpha_max=args.lr,
            alpha_min=args.lr_min,
            warmup_steps=args.warmup_steps,
            cosine_steps=args.max_iters,
        )

    # Восстановление из чекпоинта
    start_iter = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed at iteration {start_iter}")

    # Начало времени
    start_time = time.time()

    # Переменная для лучшей валидационной потери (для сохранения лучшей модели)
    best_val_loss = float('inf')

    # Основной цикл обучения
    model.train()
    iter_num = start_iter
    while iter_num < args.max_iters:
        # Получить батч
        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, args.device)
        # Прямой проход
        logits = model(inputs)
        loss = cross_entropy_loss(logits, targets)

        # Обратный проход
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)

        # Обновляем скорость обучения
        lr_current = scheduler(iter_num + 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_current

        optimizer.step()

        # --- Логирование тренировочных метрик ---
        if (iter_num + 1) % args.log_interval == 0:
            train_ppl = math.exp(loss.item())
            elapsed = time.time() - start_time
            print(f"step {iter_num+1:6d} | loss {loss.item():.4f} | ppl {train_ppl:.2f} | lr {lr_current:.2e} | time {elapsed:.1f}s")
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_perplexity": train_ppl,
                    "lr": lr_current,
                    "step": iter_num + 1,
                    "elapsed_time": elapsed,
                })

        # --- Валидация ---
        if (iter_num + 1) % args.eval_interval == 0:
            val_loss = evaluate_loss(model, val_data, args.batch_size, args.context_length, args.device)
            val_ppl = math.exp(val_loss)
            elapsed = time.time() - start_time
            print(f"VAL   step {iter_num+1:6d} | loss {val_loss:.4f} | ppl {val_ppl:.2f} | time {elapsed:.1f}s")
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val_loss": val_loss,
                    "val_perplexity": val_ppl,
                    "step": iter_num + 1,
                    "elapsed_time": elapsed,
                })

            # Запись в CSV
            train_ppl = math.exp(loss.item())   # последний train loss
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    iter_num + 1,
                    loss.item(),
                    val_loss,
                    train_ppl,
                    val_ppl,
                    lr_current,
                    elapsed,
                ])

            # Сохранение лучшей модели по валидационной потере
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join("./Checkpoints", "best_model.pt")
                save_checkpoint(model, optimizer, iter_num + 1, best_path)
                print(f"New best val loss: {val_loss:.4f}, saved to {best_path}")

        # --- Сохранение обычного чекпоинта ---
        if (iter_num + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join("./Checkpoints", f"checkpoint_{iter_num+1:07d}.pt")
            save_checkpoint(model, optimizer, iter_num + 1, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        iter_num += 1

    # Финальные сохранения
    final_path = os.path.join("./Checkpoints", "final_checkpoint.pt")
    save_checkpoint(model, optimizer, iter_num, final_path)
    print(f"Training finished. Final checkpoint saved to {final_path}")

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()