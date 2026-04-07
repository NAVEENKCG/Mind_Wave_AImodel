# 🧠 EEG Wheelchair Control: Thought-to-Command System

A production-ready deep learning system for classifying EEG signals from a TGAM (ThinkGear) module into real-world movement commands. 

The system uses a **Time-Series LSTM / CNN-LSTM hybrid** architecture to distinguish between intentional "command signals" and random "idle thoughts" with high confidence.

---

## 🛠️ Project Structure

- `config.py`: Centralized configuration for hyperparameters, serial ports, and feature mappings.
- `model.py`: PyTorch implementation of the `EEGClassifier` (LSTM and CNN-LSTM options).
- `collect_data.py`: Interactive serial data capture and labeling for training datasets.
- `preprocess.py`: Automated window generation (sliding window), normalization, and train/val/test splitting.
- `train.py`: High-performance training pipeline with learning rate scheduling and early stopping.
- `evaluate.py`: Post-training evaluation with confusion matrix heatmaps and classification reports.
- `predict_realtime.py`: Real-time inference engine that feeds live EEG data into the model and outputs hardware commands.

---

## 🚀 How to Build Your EEG Model

### 1. Hardware Connection
- Connect your TGAM module to your computer via USB-TTL or UART.
- Update `SERIAL_PORT` in `config.py` (e.g., `COM3` or `/dev/ttyUSB0`).

### 2. Phase 1: Data Collection
Collect at least 500 samples per class for robust results.
```bash
python collect_data.py
```
*Follow the prompts to perform each action (FORWARD, LEFT, RIGHT, STOP, IDLE) while wearing the headset.*

### 3. Phase 2: Preprocessing
Generate the structured windows needed for the LSTM/CNN input.
```bash
python preprocess.py
```

### 4. Phase 3: Model Training
Train the model on your collected data.
```bash
python train.py
```
*Training graphs will be saved to `training_curves.png`.*

### 5. Phase 4: Evaluation
Verify accuracy and check for class-specific confusion.
```bash
python evaluate.py
```

### 6. Phase 5: Real-time Control
Run the inference engine to begin controlling your wheelchair (or simulation).
```bash
python predict_realtime.py
```
*Commands will be sent to the wheelchair ESP32 if `WHEELCHAIR_PORT` is configured in `config.py`.*

---

## 🏗️ Architecture Details
The system processes a **sliding window of 10 timesteps (approx 10 seconds)** of EEG power bands (delta, theta, alpha, beta, gamma), attention, meditation, and blink strength.

- **Input:** `[batch, 10, 11]`
- **Optimizer:** Adam with weight decay.
- **Confidence Threshold:** 0.75 (prevents accidental triggers).

---
Created by Antigravity (AI/ML Domain Mode)
