# src/config.py

from pathlib import Path
import yaml


DEFAULT_CONFIG_PATH = "config.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    读取 config.yaml，返回一个 dict。

    这样后面 main.py、camera.py、door.py 都不用自己读 yaml，
    统一从这里拿配置。
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("配置文件为空")

    return config


def get_config_value(config: dict, key_path: str, default=None):
    """
    通过 'camera.device_index' 这种形式读取嵌套配置。

    示例：
        get_config_value(config, "camera.device_index", 0)
        get_config_value(config, "face.tolerance", 0.5)
    """
    keys = key_path.split(".")
    value = config

    for key in keys:
        if not isinstance(value, dict):
            return default

        if key not in value:
            return default

        value = value[key]

    return value