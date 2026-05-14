import os
import json
import time
import threading


class StatusStore:
    def __init__(self, state_file="runtime/state.json"):
        self.lock = threading.Lock()

        self.state_file = state_file

        self.last_name = "none"
        self.last_score = 0.0
        self.last_authorized = False
        self.last_event_time = "none"
        self.door_state = "closed"

        self._ensure_state_dir()
        self._save_to_file()

    def _ensure_state_dir(self):
        """
        确保 runtime 目录存在。
        比如 state_file = runtime/state.json，
        那就自动创建 runtime 目录。
        """
        state_dir = os.path.dirname(self.state_file)

        if state_dir:
            os.makedirs(state_dir, exist_ok=True)

    def _save_to_file(self):
        """
        把当前状态写入 JSON 文件。
        注意：这个函数默认已经在 lock 保护内调用。
        """
        data = {
            "last_name": self.last_name,
            "last_score": round(self.last_score, 3),
            "last_authorized": self.last_authorized,
            "last_event_time": self.last_event_time,
            "door_state": self.door_state,
        }

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def update_recognition(self, name, score, authorized):
        with self.lock:
            self.last_name = name
            self.last_score = float(score)
            self.last_authorized = bool(authorized)
            self.last_event_time = time.strftime("%Y-%m-%d %H:%M:%S")

            self._save_to_file()

    def set_door_state(self, state):
        with self.lock:
            self.door_state = state

            self._save_to_file()

    def to_dict(self):
        with self.lock:
            return {
                "last_name": self.last_name,
                "last_score": round(self.last_score, 3),
                "last_authorized": self.last_authorized,
                "last_event_time": self.last_event_time,
                "door_state": self.door_state,
            }