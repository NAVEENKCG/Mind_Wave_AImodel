from pathlib import Path

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

# Serial Communication
SERIAL_PORT = "COM3"
BAUD_RATE = 57600
WHEELCHAIR_PORT = "COM5"
WHEELCHAIR_BAUD = 115200

# Model Configuration
USE_CNN_LSTM = True  # Flag to switch between architectures
SEED = 42

# Data Configuration
TRAIN_WINDOW_SIZE = 10  # 1.0 seconds at 10Hz (snappy response)
INFER_WINDOW_SIZE = 10  # 1 second for faster response
STRIDE = 5              # 50% overlap for 10Hz (approx 0.5s)
N_FEATURES_RAW = 11
N_FEATURES_ENGINEERED = 18
N_CLASSES = 5
CLASS_NAMES = {
    0: "IDLE",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "STOP"
}

# Training Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
PATIENCE = 20

# Inference & Control Logic
CONFIDENCE_THRESHOLDS = {
    0: 0.50,  # IDLE
    1: 0.75,  # FORWARD
    2: 0.80,  # LEFT
    3: 0.80,  # RIGHT
    4: 0.65   # STOP
}
HOLD_REQUIRED = 3  # Successive predictions needed
COOLDOWN_SECONDS = {
    1: 2.0,   # FORWARD
    2: 3.0,   # LEFT
    3: 3.0,   # RIGHT
    4: 1.0    # STOP
}

# Quality Filter Thresholds (during collection)
QUALITY_FILTERS = {
    1: {"attention": 65},
    2: {"theta_dominant": True},
    3: {"alpha_dominant": True},
    4: {"meditation": 65},
    0: {"no_dominant": True}
}

# Final File Pathing for inference
MODEL_PATH = MODELS_DIR / "best_model.pth"
COMMANDS = CLASS_NAMES
