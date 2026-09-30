
import os

import numpy as np
from cs336_basics.Modules.BPETokenizer import BPETokenizer



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, "Datasets", "TinyStories-Train.pickle")
    vocab_path = os.path.abspath(vocab_path)  # превращает в нормализованный абсолютный путь

    special_tokens = ["<|endoftext|>"] 

    tokenizer = BPETokenizer.from_files(vocab_path, vocab_path, special_tokens)

    # Чтение файла
    input_file = os.path.join(script_dir, "Datasets", "TinyStories-valid-mini.txt")
    input_file = os.path.abspath(input_file)  # превращает в нормализованный абсолютный путь

    all_tokens = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = tokenizer.encode(line)
                all_tokens.extend(tokens)
            # если есть разделитель, можно добавить специальный токен
            # all_tokens.append(tokenizer.inverted_vocab["<|endoftext|>".encode()])

    # Сохраняем в .npy

    save_path = os.path.join(script_dir, "Datasets", "TinyStories-valid-mini.npy")
    save_path = os.path.abspath(save_path)  # превращает в нормализованный абсолютный путь
    np.save(save_path, np.array(all_tokens, dtype=np.int32))

if __name__ == "__main__":
    main()