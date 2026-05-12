import os
import time


class AccessLogger:
    def __init__(self, log_path="data/logs/access.log",unknown_cooldown=3.0):
        self.log_path = log_path
        self.unknown_cooldown = unknown_cooldown
        self.last_unknown_log_time = 0

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def write(self, name, score, authorized):
        if not authorized:
            now_time = time.time()

            if now_time - self.last_unknown_log_time < self.unknown_cooldown:
                return False

            self.last_unknown_log_time = now_time

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        result = "AUTHORIZED" if authorized else "UNKNOWN"

        line = f"{now}, {result}, name={name}, score={score:.3f}\n"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

        print("[LOG]", line.strip())