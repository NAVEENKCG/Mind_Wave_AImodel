import torch
import os

# --- Hardware / Serial Settings ---
SERIAL_PORT = "COM3"  # Update to your TGAM COM port (e.g., COM3 on Win, /dev/ttyUSB0 on Linux)
BAUD_RATE = 57600
WHEELCHAIR_PORT = None  # COM port for wheelchair ESP32 if sending commands

# --- Data Settings ---
FEATURES = [
    "delta", "theta", "lowAlpha", "highAlpha", "lowBeta", 
    "highBeta", "lowGamma", "highGamma", "attention", "meditation", "blink"
]
NUM_FEATURES = len(FEATURES)
CLASSES = {
    0: "IDLE",
    1: "FORWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "STOP"
}
NUM_CLASSES = len(CLASSES)

# --- Preprocessing ---
WINDOW_SIZE = 10  # 10 samples per window
STRIDE = 1
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# --- Model Architecture ---
USE_CNN_LSTM = False  # Toggle between LSTM and CNN-LSTM
HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32
DROPOUT = 0.3

# --- Training Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SEED = 42

# --- Real-time Settings ---
CONFIDENCE_THRESHOLD = 0.75
PREDICTION_INTERVAL = 2.0  # Seconds between predictions

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw_data.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
