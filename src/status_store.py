import time
import threading

class StatusStore:
    def __init__(self):
        self.lock = threading.Lock()

        self.last_name = "none"
        self.last_score = 0.0
        self.last_authorized = False
        self.last_event_time = "none"
        self.door_state = "closed"

    def update_recognition(self, name, score, authorized):
        with self.lock:
            self.last_name = name
            self.last_score = float(score)
            self.last_authorized = bool(authorized)
            self.last_event_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def set_door_state(self, state):
        with self.lock:
            self.door_state = state

    def to_dict(self):
        with self.lock:
            return {
                "last_name": self.last_name,
                "last_score": round(self.last_score, 3),
                "last_authorized": self.last_authorized,
                "last_event_time": self.last_event_time,
                "door_state": self.door_state,
	    }
