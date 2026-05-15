# 🧠 ORBIT AI — Universal BCI System (v2.0)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow)

## 🎥 Demo
![Demo](demo.gif)
*Real-time prediction running with TGAM module and Riemannian geometry pipeline.*

---

ORBIT AI is a state-of-the-art **Brain-Computer Interface** designed for medical wheelchair control. It uses a **Hybrid BCI-EMG** architecture that fuses clinical-grade neural decoding (EEG) with high-speed muscle signal detection (EMG) for maximum safety.

---

## 📊 Results

Model achieves 87% accuracy on IDLE vs FORWARD classification using Riemannian geometry pipeline.

---

## 🛠️ Technology Stack

- **Hardware:** BioAmp EXG Pill (Neuro-Analog Front-End) + Arduino/ESP32.
- **AI Core:** **MOABB** (Mother of All BCI Benchmarks).
- **Processing:** Riemannian Geometry (Tangent Space Mapping + Covariance Estimation).
- **Safety:** Hybrid Jaw-Clench Stop + 5-Layer Security Pipeline.
- **Interface:** Real-time TUI (Terminal User Interface) built with Python `Rich`.

---

## 📸 Hardware Setup

![Hardware Setup](hardware_setup.jpg)
*ESP32 + Electrodes running the ORBIT AI data acquisition pipeline.*

---

## 📐 Hybrid Command Mapping

We use a "Safety-First" mapping strategy:

| Command | Signal Type | Mental / Physical Task | Brain Bio-Marker |
| :--- | :--- | :--- | :--- |
| **FORWARD** | 🧠 EEG | Imagine Moving Both Feet | Cz Central Power |
| **LEFT** | 🧠 EEG | Imagine Squeezing Left Hand | C4 Alpha/Beta shift |
| **RIGHT** | 🧠 EEG | Imagine Squeezing Right Hand | C3 Alpha/Beta shift |
| **IDLE** | 🧠 EEG | Relaxed state, eyes open | Baseline Baseline |
| **STOP** | 💪 EMG | **Jaw Clench (Bite Teeth)** | High-Frequency Spike |

---

## 🚀 Getting Started

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. The "Universal Brain" Training
Instead of training from scratch, we use MOABB to learn from 100+ clinical EEG subjects (PhysioNet + BNCI datasets).
```powershell
python train_moabb.py
```
*This script automatically searches for the best engine (MDM vs LDA vs SVM) for your data.*

### 3. Personal Calibration (Personalization)
Every brain is unique. Run this 45-second session to map your personal bio-thresholds.
```powershell
python calibrate.py
```

### 4. Real-time Dashboard
Launch the dashboard with the simulator (for testing) or the BioAmp hardware.
```powershell
# For Simulator
python simulate_tgam.py
python predict_realtime.py

# For Real Hardware
python bridge_bioamp.py
python predict_realtime.py
```

---

## 🛡️ 5-Layer Safety Pipeline
The `predict_realtime.py` engine implements:
1.  **Signal Quality Gate:** Rejects commands if electrodes have poor contact.
2.  **Warmup Gate:** 2-minute "Brain-Settle" period required before movement.
3.  **Fatigue Monitor:** Auto-slows the chair if drowsiness (Theta waves) is detected.
4.  **EMG Emergency Stop:** Jaw clench stops the chair in < 100ms.
5.  **Weighted Voting:** Smooths movement by voting on the last 3 predictions.

---

## 🗺️ System Architecture

![System Architecture](architecture.png)

For a deep dive into the code structure and data pipelines, see [ARCHITECTURE.md](./ARCHITECTURE.md).
