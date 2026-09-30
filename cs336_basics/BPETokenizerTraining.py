import argparse
from datetime import datetime
import os
import pickle
import sys
import time

from cs336_basics.Modules import BPETokenizer

def main(args: list[str]):
    parser = argparse.ArgumentParser("encode data")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "Datasets", "TinyStories-train.txt")
    dataset_path = os.path.abspath(dataset_path)  # превращает в нормализованный абсолютный путь
    parser.add_argument("--vocab")

    config = parser.parse_args(args)

    result = BPETokenizer.train(dataset_path, int(config.vocab), ["<|endoftext|>"])
    
    print(result)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_name = "TinyStories-Train"
    output_path = os.path.join(script_dir, "Datasets", f"{file_name}-{time.time()}.pickle")
    output_path = os.path.abspath(output_path)

    # Создаём директорию, если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(result, f)
    


if __name__ == "__main__":
    main(sys.argv[1:])
    


