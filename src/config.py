import os
import yaml

# Dynamically calculate the absolute route pointing to the config.yaml located in the project Root limit structurally!
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing configuration constraints! Did not find: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Expose structurally generic configuration branches mapped directly:
PATHS = CONFIG.get("paths", {})
TRAINING = CONFIG.get("training", {})
MODEL = CONFIG.get("model", {})
DATASET = CONFIG.get("dataset", {})
