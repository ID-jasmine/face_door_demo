import time
from pathlib import Path

class MockGPIO:
    """
    电脑端 GPIO 模拟层。
    后面上 Orange Pi 后，可以把这里替换成真实 GPIO 控制。
    """

    def set_high(self):
        print("[GPIO MOCK] GPIO HIGH")

    def set_low(self):
        print("[GPIO MOCK] GPIO LOW")

class LedGPIO:
    """
    Orange Pi 板载 status_led 控制层。
    用 /sys/class/leds/status_led 模拟开锁输出。
    """

    def __init__(self, led_name="status_led", active_high=True):
        self.led_name = led_name
        self.active_high = active_high

        self.led_dir = Path("/sys/class/leds") / led_name
        self.trigger_path = self.led_dir / "trigger"
        self.brightness_path = self.led_dir / "brightness"

        if not self.led_dir.exists():
            raise RuntimeError(f"LED not found: {self.led_dir}")

        # 关闭系统自动闪烁，改为手动控制
        self.trigger_path.write_text("none")

    def set_high(self):
        value = "1" if self.active_high else "0"
        self.brightness_path.write_text(value)
        print("[LED GPIO] LED ON")

    def set_low(self):
        value = "0" if self.active_high else "1"
        self.brightness_path.write_text(value)
        print("[LED GPIO] LED OFF")

class AccessController:
    def __init__(self, gpio=None, cooldown=3.0):
        self.gpio = gpio if gpio is not None else MockGPIO()
        self.cooldown = cooldown
        self.last_open_time = 0
        self.is_on = False

    def set_authorized(self, authorized: bool):
        if authorized:
            if not self.is_on:
                print("[ACCESS] AUTHORIZED -> LED ON")
                self.gpio.set_high()
                self.is_on = True
            return True
        else:
            if self.is_on:
                print("[ACCESS] NOT AUTHORIZED -> LED OFF")
                self.gpio.set_low()
                self.is_on = False
            return False

    def open_door(self, name, score):
        """
        保留旧接口：识别成功时常亮，不再自动熄灭。
        """
        print(f"[ACCESS] AUTHORIZED: {name}, score={score:.3f}")
        return self.set_authorized(True)

    def close_door(self):
        return self.set_authorized(False)