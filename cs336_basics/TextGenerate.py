import os
import torch
import torch.nn.functional as F
from cs336_basics.Modules.BPETokenizer import BPETokenizer
from cs336_basics.Modules.Functions import generate
from cs336_basics.Modules.TransformerLM import TransformerLanguageModel

# ---------- ПАРАМЕТРЫ (ДОЛЖНЫ СОВПАДАТЬ С ОБУЧЕНИЕМ) ----------
VOCAB_SIZE = 10000
D_MODEL = 512
CONTEXT_LENGTH = 256
NUM_HEADS = 16
NUM_LAYERS = 4
D_FF = 1344          # автоматически 4*d_model = 2048? Нет, (8/3)*512 ≈ 1365, кратно 64 → 1408
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- ЗАГРУЗКА МОДЕЛИ ----------
def load_model(path):
    model = TransformerLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        context_length=CONTEXT_LENGTH,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        theta=10000.0,
        eps=1e-5,
        mask=True,
        device=DEVICE,
        dtype=torch.float32,
    )
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    print(f"Модель загружена, параметров: {sum(p.numel() for p in model.parameters()):,}")
    return model

# ---------- ИНТЕРАКТИВНЫЙ РЕЖИМ ----------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, "Datasets", "TinyStories-Train.pickle")
    checkpoint_path = os.path.join(script_dir, "Checkpoints", "best_model.pt") 

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer.from_files(vocab_path, vocab_path, special_tokens)
    model = load_model(checkpoint_path)

    print("Инициализация. Введите текст (или 'exit'):")
    while True:
        prompt = input("\n> ")
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt.strip():
            continue

        # Токенизация (возвращает список int)
        token_ids = tokenizer.encode(prompt)   # list[int]

        # Преобразование в тензор (1, seq_len)
        prompt_tensor = torch.tensor(token_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # Генерация
        stop_id = tokenizer.inverted_vocab.get("<|endoftext|>".encode("UTF-8"))
        output_tensor = generate(
            model,
            prompt_tensor,
            max_new_tokens=120,
            temperature=0.8,
            top_k=None,
            top_p=0.9,
            stop_token_id=stop_id
        )

        # Декодирование: output_tensor имеет размер (1, total_len)
        output_tokens = output_tensor[0].tolist()   # список int

        print("Special tokens:", tokenizer.special_tokens)
        print("Inverted vocab keys (первые 10):", list(tokenizer.inverted_vocab.keys())[:10])
        print("ID для '<|endoftext|>':", tokenizer.encode("<|endoftext|>"))
        print("Stop token_id:", stop_id)
        print("Generated tokens:", output_tensor[0].tolist())
        output_text = tokenizer.decode(output_tokens)
        print("\n" + output_text)

if __name__ == "__main__":
    main()