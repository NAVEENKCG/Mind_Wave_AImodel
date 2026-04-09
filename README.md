# ORBIT AI — Brain-Computer Interface (BCI) System

ORBIT AI is a high-performance EEG-based control system designed to classify neural brain states into wheelchair movement commands. Using a single-channel TGAM (ThinkGear) module and a hybrid CNN-LSTM neural network, ORBIT AI translates specific mental tasks into physical navigation.

## 🛠 Project Components

- **EEG Hardware:** TGAM module (Fp1 position).
- **MCU:** ESP32 for signal relay (UART to PC) and wheelchair control.
- **AI Core:** PyTorch-based Deep Learning (CNN + Bi-LSTM + Self-Attention).
- **Interface:** Real-time dashboard using Python `Rich`.

## 📐 Wiring Diagram (Text-based)

```text
TGAM Module (EEG)          ESP32 (Data)          PC (Python)          ESP32 (Wheelchair)
[TX] -------------------> [RX] (UART)
[VCC/GND] <-------------> [3.3V/GND]
                          [USB] -------------> [Serial COM3]
                                               [Serial COM5] -------> [RX] (UART)
                                                                      [GPIOs] ------> [Motor Driver]
```

## 🧠 Class Definitions & Mental Tasks

| Class | Command | Mental Task | Signal Pattern |
| :--- | :--- | :--- | :--- |
| **0** | **IDLE** | Relaxed awake state, no task. | Moderate bands, no dominance. |
| **1** | **FORWARD** | Intense focus (Serial counting 300-3). | High Beta, Attention > 75. |
| **2** | **LEFT** | Arithmetic working memory (500-7). | High Theta, Theta/Beta ratio ^. |
| **3** | **RIGHT** | Spatial imagination (Walk through home). | High Alpha, focus on colors/depth. |
| **4** | **STOP** | Deep meditation, clear mind. | High Alpha/Theta, Meditation > 75. |

## 🚀 Getting Started

### 1. Installation

```powershell
pip install -r requirements.txt
```

### 2. Data Collection

Collect at least 100 samples per class using the strict protocol:

```powershell
python collect_data.py
```

*Follow the 10s preparation -> 5s transition -> 3s collection window.*

### 3. Preprocessing

Run the 8-step pipeline including derived features and augmentation (1 real sample → 9 samples):

```powershell
python preprocess.py
```

### 4. Training

Train the CNN-LSTM hybrid model with Cosine Annealing:

```powershell
python train.py
```

### 5. Evaluation

Check accuracy and safety metrics (F1 scores, confusion matrix):

```powershell
python evaluate.py
```

### 6. Real-time Inference

Launch the live dashboard and control the wheelchair:

```powershell
python predict_realtime.py
```

## 🛡 Performance Benchmarks (Target)

- **IDLE Accuracy:** > 85% (Safety critical)
- **STOP Response:** < 1.0s (Interrupt priority)
- **False Positives:** < 5% from IDLE state.

## ⚠️ Troubleshooting

- **No Data:** Check if Serial port `COM3` is correct in `config.py`.
- **Poor Signal:** Ensure the forehead sensor (Fp1) is clean and the ear clip is secure.
- **Inconsistent Control:** Recalibrate your mental states; ensure 15s rest between triggers.
