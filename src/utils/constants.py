from pathlib import Path

# path constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# prediction constants
IMAGE_SIZE = [224, 224]
CLASS_MAP = {
    0: "Normal",
    1: "Pneumonia"
}