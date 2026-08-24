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
TRAIN_WINDOW_SIZE = 8   # 0.8 seconds at 10Hz (snappier response)
INFER_WINDOW_SIZE = 8   # 0.8 seconds for faster response
STRIDE = 5              # 50% overlap for 10Hz (approx 0.5s)
N_FEATURES_RAW = 8      # 5 bands + attention + meditation + blink
N_FEATURES_ENGINEERED = 18
N_CLASSES = 5
N_CLASSES_MOABB = 2
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

# EMG / ERD Configuration
EMG_THRESHOLD_UV = 500.0
ERD_THRESHOLD = 0.15
ERD_SYMMETRY_TOLERANCE = 0.30

# Motor Imagery Channel Definitions
MOTOR_CHANNEL_NAMES = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
]
N_MOTOR_CHANNELS = len(MOTOR_CHANNEL_NAMES)
C3_INDEX = MOTOR_CHANNEL_NAMES.index('C3..')
C4_INDEX = MOTOR_CHANNEL_NAMES.index('C4..')

# ── Adaptive Drift Engine (Model 2) ─────────────────────────────
# The drift engine NEVER modifies moabb_csp_lda.pkl (Model 1).
# It only writes to drift_profile.json with threshold offsets.
ADAPTATION_ENABLED = True
DRIFT_PROFILE_PATH = MODELS_DIR / "drift_profile.json"
MIN_SESSIONS_FOR_ADAPTATION = 5    # Need at least 5 valid sessions
MIN_EVENTS_PER_SESSION = 100       # Sessions with <100 events are skipped
DRIFT_EWMA_ALPHA = 0.3             # EWMA smoothing factor (higher = more weight to recent)
