import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

LOGS_DIR.mkdir(exist_ok=True)

# Artifact file paths
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(ARTIFACTS_DIR / "battery_life_model.pkl")))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(ARTIFACTS_DIR / "scaler.pkl")))
LABEL_ENCODER_PATH = Path(os.getenv("LABEL_ENCODER_PATH", str(ARTIFACTS_DIR / "label_encoder.pkl")))

# Model expects these exact class labels from training
VALID_CYCLE_TYPES = ("charge", "discharge", "impedance")

# Feature order must match training pipeline
FEATURE_ORDER = ["type", "Capacity", "Re", "Rct"]

# Streamlit
APP_TITLE = "EV Battery Life Prediction"
APP_ICON = "⚡"
PAGE_LAYOUT = "wide"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "app.log"
