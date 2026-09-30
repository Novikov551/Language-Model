import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path=None, save_plots=True, show_plots=False):
    """
    Строит графики loss и perplexity из CSV-файла метрик.
    
    Args:
        csv_path: путь к metrics.csv (по умолчанию ищет в ../Metrics/metrics.csv)
        save_plots: сохранять ли графики в файлы PNG
        show_plots: показывать ли графики интерактивно
    """
    if csv_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Metrics", "metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден. Сначала обучите модель.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} записей")
    
    # Создаём два графика
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 1. Loss
    ax1.plot(df["step"], df["train_loss"], label="Train Loss", marker='.', alpha=0.7)
    ax1.plot(df["step"], df["val_loss"], label="Validation Loss", marker='.', alpha=0.7)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Perplexity (если есть столбцы)
    if "train_ppl" in df.columns and "val_ppl" in df.columns:
        ax2.plot(df["step"], df["train_ppl"], label="Train Perplexity", marker='.', alpha=0.7)
        ax2.plot(df["step"], df["val_ppl"], label="Validation Perplexity", marker='.', alpha=0.7)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("Training and Validation Perplexity")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "Perplexity data not available", 
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Perplexity")
    
    plt.tight_layout()
    
    if save_plots:
        base = os.path.splitext(csv_path)[0]
        loss_plot = os.path.join(os.path.dirname(csv_path), "loss_curve.png")
        ppl_plot = os.path.join(os.path.dirname(csv_path), "perplexity_curve.png")
        plt.savefig(loss_plot, dpi=150, bbox_inches='tight')
        print(f"График loss сохранён: {loss_plot}")
        # Сохраняем отдельно график perplexity? Сейчас в одном файле два subplot.
        # Для удобства можно сохранить оба в одном файле.
        plt.savefig(os.path.join(os.path.dirname(csv_path), "metrics_combined.png"), dpi=150)
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--csv", type=str, default=None, help="Path to metrics.csv")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save plots to files")
    args = parser.parse_args()
    
    plot_metrics(
        csv_path=args.csv,
        save_plots=not args.no_save,
        show_plots=args.show
    )

if __name__ == "__main__":
    main()