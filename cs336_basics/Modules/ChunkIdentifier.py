import logging
import os

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class ChunkIdentifier:
    def __init__(self, path: str, delimeter: str) -> None:
        if not os.path.exists(path):
            raise Exception(f"File {path} was not found")

        self.path = path
        self.delimeter = delimeter

    def get_chunks_positions(self, num_workers: int = 0) -> list[tuple[int, int]]:
        chunks_positions: list[tuple[int, int]] = []

        if not num_workers:
            cores = os.cpu_count()
            num_workers = cores if cores else 1

        if num_workers < 3:
            log.warning(f"Num of workers is low: {num_workers}")
        else:
            log.info(f"Using {num_workers} workers")

        file_size = os.path.getsize(self.path)

        if num_workers > file_size:
            raise Exception(f"Chunking file with size {file_size}B is not supported among {num_workers} workers")

        chunk_len = int(file_size / num_workers)

        slide_len = min(int(chunk_len / num_workers), file_size - chunk_len)
        delimeter_bytes = self.delimeter.encode("UTF-8")

        with open(self.path, "rb+") as file:
            start_position = 0
            for _ in range(num_workers):
                chunk_end = start_position + chunk_len
                file.seek(chunk_end)

                contents = b""
                num_slides = 0
                while delimeter_bytes not in contents:
                    if file.tell() + chunk_len > file_size:
                        chunks_positions.append((start_position, file_size))
                        return chunks_positions

                    num_slides += 1
                    contents = file.read(slide_len)

                position = contents.find(delimeter_bytes)
                chunk_end += ((num_slides - 1) * slide_len) + position + len(delimeter_bytes)
                chunks_positions.append((start_position, chunk_end))
                start_position = chunk_end

        return chunks_positions