import time


class MockGPIO:
    """
    电脑端 GPIO 模拟层。
    后面上 Orange Pi 后，可以把这里替换成真实 GPIO 控制。
    """

    def set_high(self):
        print("[GPIO MOCK] GPIO HIGH")

    def set_low(self):
        print("[GPIO MOCK] GPIO LOW")


class AccessController:
    def __init__(self, gpio=None, open_duration=1.0, cooldown=3.0):
        self.gpio = gpio if gpio is not None else MockGPIO()
        self.open_duration = open_duration
        self.cooldown = cooldown
        self.last_open_time = 0

    def can_open(self):
        now = time.time()
        return now - self.last_open_time >= self.cooldown

    def open_door(self, name, score):
        if not self.can_open():
            return False

        self.last_open_time = time.time()

        print(f"[ACCESS] OPEN DOOR: {name}, score={score:.3f}")

        self.gpio.set_high()
        time.sleep(self.open_duration)
        self.gpio.set_low()

        return True
    