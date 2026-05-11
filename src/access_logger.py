import os
import time


class AccessLogger:
    def __init__(self, log_path="data/logs/access.log"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def write(self, name, score, authorized):
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        result = "AUTHORIZED" if authorized else "UNKNOWN"

        line = f"{now}, {result}, name={name}, score={score:.3f}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

        print("[LOG]", line.strip())