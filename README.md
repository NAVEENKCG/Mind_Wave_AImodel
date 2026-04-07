# 🧠 MindWave AI: EEG Wheelchair Control System

---

## 🎯 Project Goal
The primary objective of **MindWave AI** is to restore independence and mobility to individuals with severe motor impairments (such as quadriplegia or ALS) by developing a high-fidelity, cognitive-driven robotic control system. Our goal is to translate raw brain activity into reliable locomotion commands with over 90% accuracy, bridging the gap between intention and action.

## 📝 Executive Summary
MindWave AI is an end-to-end Brain-Computer Interface (BCI) solution that leverages the **ThinkGear AM (TGAM)** EEG chipset to capture neurofeedback. Unlike traditional threshold-based systems, our platform utilizes a customized **Time-Series CNN-Bi-LSTM + Attention** hybrid model to analyze real-time brainwave power bands (Delta, Alpha, Beta, etc.). This allow for:
- **Intention Classification**: Distinguishing between relax-state (IDLE) and active command focus (FORWARD).
- **Artifact Recognition**: Mapping physical blinks to specific directional commands (LEFT, RIGHT, STOP).
- **Edge Deployment**: Real-time inference on standard local hardware, communicating with a wheelchair's ESP32 unit via serial protocols.

## 📊 Business Analytics & Market Impact

### 1. Market Opportunity
- **Accessibility Gap**: Over 75 million people worldwide require a wheelchair, but high-end motorized systems often lack intuitive control for those with minimal upper-body strength.
- **Cost Efficiency**: Medical-grade EEG systems cost upwards of $10,000. MindWave AI provides a "frugal innovation" alternative using consumer-grade sensors and robust software, reducing entry barriers by **80-90%**.

### 2. Strategic Impact (KPIs)
- **Response Latency**: Targeted at <1.5s for safe real-time maneuverability.
- **Confidence Threshold**: Operating at a **0.75 threshold** to minimize False Positives, ensuring user safety in dense environments.
- **Scalability**: While currently configured for wheelchairs, the modular Bi-LSTM architecture can be adapted for smart home automation, surgical robotics, or non-verbal communication tools.

### 3. Sustainability & Growth
The system features a centralized **Config-as-Code** model, allowing for rapid deployment across different hardware modules without rewriting the core classification logic. By open-sourcing the training pipeline and interactive notebooks, we accelerate community-driven improvements in EEG signal processing.

---

# 🧠 EEG Wheelchair Control: Thought-to-Command System

A production-ready deep learning system for classifying EEG signals from a TGAM (ThinkGear) module into real-world movement commands. 

The system uses a **Time-Series LSTM / CNN-LSTM hybrid** architecture to distinguish between intentional "command signals" and random "idle thoughts" with high confidence.

---

#### 🛠️ Project Structure

- `config.py`: Centralized configuration for hyperparameters, serial ports, and feature mappings.
- `model.py`: PyTorch implementation of the `EEGClassifier` (Advanced CNN-Bi-LSTM + Attention).
- `collect_data.py`: Interactive serial data capture and labeling for training datasets.
- `preprocess.py`: Automated window generation (sliding window), normalization, and train/val/test splitting.
- `train.py`: High-performance training pipeline with learning rate scheduling and early stopping.
- `evaluate.py`: Post-training evaluation with confusion matrix heatmaps and classification reports.
- `predict_realtime.py`: Real-time inference engine that feeds live EEG data into the model and outputs hardware commands.
- `notebooks/`: **(NEW)** Interactive Jupyter Notebooks for project development:
  - `01_data_exploration.ipynb`: Visualize raw EEG signals and feature distributions.
  - `02_model_experiments.ipynb`: Prototype and compare different model architectures.
  - `03_results_analysis.ipynb`: Deep-dive into model errors and confusion matrices.

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
