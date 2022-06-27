import os
from pathlib import Path
from .utils import load_cfg

PROJECT_DIR = Path(__file__).parent.parent
CONFIG_PATH = os.environ.get("CONFIG_PATH", PROJECT_DIR / "config.yaml")
CONFIG = load_cfg(CONFIG_PATH)
