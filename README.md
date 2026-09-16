# 🧠 ORBIT AI — Universal BCI System (v3.1)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyRiemann](https://img.shields.io/badge/PyRiemann-Riemannian_BCI-orange)
![MOABB](https://img.shields.io/badge/MOABB-1.1-green)
![Stage](https://img.shields.io/badge/Stage-4--Class%20Motor%20Imagery%20%2B%20Dual--Model-blueviolet)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**ORBIT AI** is an open-source **Brain-Computer Interface** designed for medical prosthetic arm control. It uses a **Hybrid BCI-EMG** architecture — EEG-based neural decoding for multi-class movement commands, and EMG-based jaw-clench detection for instant emergency stop.

> **4-Class Motor Imagery** (IDLE, LEFT, RIGHT, FORWARD/LIFT) via Riemannian FBCSP+LDA · **Model 2** Adaptive Neural Drift Engine · **< 100ms** emergency stop via jaw-clench EMG · **95-second** personal calibration

---

## 💡 Product Ideology

### The Problem We Refuse to Ignore

Approximately **5.4 million people** in the United States alone live with some form of paralysis. Worldwide, hundreds of millions of individuals with conditions like ALS, spinal cord injuries, cerebral palsy, and locked-in syndrome are denied the most basic human freedom — **the freedom to move**.

Traditional assistive technologies — joysticks, sip-and-puff controllers, eye-tracking systems — all demand some level of residual physical ability. For those with the most severe impairments, these solutions simply **do not exist**. The mind remains sharp, but the body cannot respond.

**ORBIT AI exists because thought should be enough.**

### Our Vision

> *A world where no conscious mind is trapped in a motionless body — where the mere intention to move is sufficient to move.*

### Our Mission

To build an **open-source, medically safe, and universally accessible** Brain-Computer Interface that translates raw neural signals into reliable prosthetic arm commands — and to make it available to **everyone**, not just those who can afford proprietary clinical systems costing $50,000+.

### Core Design Principles

| Principle | What It Means in Practice |
| :--- | :--- |
| 🛡️ **Safety Above All** | No prosthetic arm command is ever issued without passing a 5-layer security pipeline. A single jaw clench halts everything in under 100ms. The system would rather **do nothing** than do the wrong thing. |
| ♿ **Radical Accessibility** | The entire system costs under **₹5,000** (~$60) in hardware. The software is MIT-licensed and runs on consumer laptops. No hospital visit required. No proprietary lock-in. |
| 🧬 **Open Science, Open Code** | Every algorithm, every dataset reference, every training pipeline is public. We publish our research paper ([RND_PAPER.md](./RND_PAPER.md)) alongside our code — because breakthroughs locked behind paywalls don't save lives. |
| 🧠 **The Brain Adapts, So Must We** | No two brains are identical. ORBIT AI trains on a "Universal Brain" model from clinical datasets, then personalizes through a 95-second calibration. The system adapts to **you**, not the other way around. |
| 🤝 **Human Dignity First** | This is not a tech demo. Every design decision — from the fatigue monitor that slows prosthetic arm motion when you're tired, to the warm-up gate that waits until your brain signals stabilize — treats the user as a **person**, not a data source. |
| ⚡ **Hybrid Intelligence** | We don't rely on a single signal. EEG provides the intention. EMG provides the emergency brake. AI provides the pattern recognition. Redundancy is not overhead — it is **survival engineering**. |

---

## 📐 Hybrid Command Mapping

We use a **"Safety-First"** dual-modality strategy — mental tasks for direction, jaw clench for instant stop:

| Command | Signal | Mental / Physical Task | Brain Bio-Marker |
| :--- | :--- | :--- | :--- |
| **LIFT / FORWARD** | 🧠 EEG | Imagine moving both feet | Cz Central Beta Power ↑ |
| **LEFT** | 🧠 EEG | Imagine squeezing left hand | C4 Alpha/Beta shift |
| **RIGHT** | 🧠 EEG | Imagine squeezing right hand | C3 Alpha/Beta shift |
| **IDLE** | 🧠 EEG | Relaxed state, eyes open | Alpha power dominance |
| **STOP** | 💪 EMG | **Jaw clench (bite teeth)** | High-frequency spike >100μV |

> **Current Stage:** The active Riemannian ML pipeline (`train_moabb.py`), live TUI (`predict_realtime.py`), and WebSocket server (`predict_websocket.py`) classify all **4 motor imagery classes** (`IDLE`, `LEFT`, `RIGHT`, `FORWARD`/`LIFT`). Hardware emergency stop (`STOP`) is handled via fast-path EMG jaw-clench detection in under 100ms.

---

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+ and npm (only required for the web dashboard)
- For real hardware mode: BioAmp EXG Pill connected via USB, 
  update `SERIAL_PORT` in `config.py`

---

## ⚡ Quick Start (Demo in 2 commands)

Terminal 1:
```bash
python simulate_tgam.py
```

Terminal 2:
```bash
python predict_realtime.py --demo
```

OR one command:
```bash
python demo_mode.py
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Hardware** | BioAmp EXG Pill (Gain ~1000×) / AD8232 (budget alt.) + ESP32/Arduino |
| **Signal Processing** | MNE-Python (FIR filtering 0.5–50 Hz) + SciPy (Welch PSD, Butterworth filters) |
| **AI Core — Active** | 4-Class PyRiemann (Covariances → Tangent Space → LDA) with FBCSP 5-band feature extraction |
| **AI Core — Adaptive** | Adaptive Neural Drift Engine (Model 2) — offline drift detection & threshold auto-adjustment |
| **AI Core — Legacy** | PyTorch CNN-LSTM (Conv1D → BiLSTM → FC) — **not used in live inference** |
| **Safety** | 5-Layer Gate Pipeline (Signal Quality · Warm-Up · Fatigue · EMG Stop · Voting) |
| **Interface — Terminal** | Real-time TUI dashboard with Python `Rich` (4-class display) |
| **Interface — Web** | Next.js 15 Prosthetic Arm Simulator (`prosthetic-arm-simulator/`, port 3001) & Wheelchair Simulator (`wheelchair-simulator/`, port 3000) via WebSocket |
| **Diagnostics** | Session logging (100ms telemetry), auto-generated reports, drift profiling |

---

## 🗺️ System Architecture

```mermaid
flowchart TD
    subgraph T1["Tier 1 — Signal Acquisition"]
        EEG["🧠 EEG"] --> HW["BioAmp EXG Pill"]
        EMG["💪 EMG"] --> HW
        HW --> MCU["ESP32 / Arduino"]
    end

    subgraph T2["Tier 2 — Feature Extraction & Classification"]
        COV["Covariances (LWF)"] --> TS["Tangent Space"]
        TS --> CLS["Classifier (LDA/SVM/MDM)"]
    end

    subgraph T3["Tier 3 — Safety & Control"]
        GATE["5-Layer Safety Gate"]
        GATE --> TUI["TUI Dashboard"]
        GATE --> WC["Prosthetic Arm Action"]
    end

    MCU -->|"USB Serial @ 115200"| COV
    CLS --> GATE

    style GATE fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
```

> 📄 For the full research paper with references, see [RND_PAPER.md](./RND_PAPER.md)

---

## 🧬 What Algorithm Does ORBIT AI Currently Use?

### The Active Pipeline: Riemannian FBCSP + LDA

ORBIT AI's **live inference system** (`predict_realtime.py` and `predict_websocket.py`) uses a **Riemannian Geometry** pipeline — not a neural network. This is the gold standard for motor imagery BCI because it generalizes across different users without retraining.

**How it works — step by step:**

```
Raw EEG (21 motor-cortex channels × 160 samples = 1 second)
        │
        ▼
┌────────────────────────────────────────────┐
│  1. FILTER BANK (FBCSP)                    │
│     Splits signal into 5 frequency bands:  │
│     δ (0.5–4 Hz), θ (4–8), α (8–13),      │
│     β (13–30), γ (30–50)                   │
└──────────────────┬─────────────────────────┘
                   ▼
┌────────────────────────────────────────────┐
│  2. COVARIANCE MATRICES                    │
│     Computes how each channel relates to   │
│     every other (Ledoit-Wolf shrinkage)    │
│     → 21×21 matrix per band                │
└──────────────────┬─────────────────────────┘
                   ▼
┌────────────────────────────────────────────┐
│  3. RIEMANNIAN TANGENT SPACE               │
│     Projects curved manifold data onto     │
│     flat Euclidean space for classification │
└──────────────────┬─────────────────────────┘
                   ▼
┌────────────────────────────────────────────┐
│  4. MULTI-CLASS LDA CLASSIFIER             │
│     Separates IDLE, LEFT, RIGHT, FORWARD   │
│     + Platt-calibrated probability output  │
└──────────────────┬─────────────────────────┘
                   ▼
      IDLE / LEFT / RIGHT / FORWARD (LIFT)
              (with confidence %)
```

**Why Riemannian over CNN-LSTM?**

| Factor | Riemannian (active) | CNN-LSTM (legacy) |
|--------|-------------------|-------------------|
| **Cross-user generalization** | ✅ Excellent — works on new users | ❌ Overfits to training subjects |
| **Calibration needed** | 95-second profile | Hours of personal data |
| **Validated accuracy** | **83.71%** (binary baseline) / 4-Class active | **Untested** (never formally evaluated) |
| **What it looks at** | Channel-to-channel relationships | Raw signal values over time |
| **Inference speed** | Very fast (matrix math) | Slower (neural network forward pass) |

**Key training details** (in `train_moabb.py`):
- **Dataset:** PhysioNet EEGMMIDB — 10 subjects, 21 motor-cortex channels (sensorimotor strip: FC5–FC6, C5–C6, CP5–CP6)
- **Motor imagery & rest mapping:**
  - **Rest baseline runs:** R01, R02 → `IDLE` (0)
  - **Left fist runs:** R03, R07, R11, R05, R09, R13 (T1 events) → `LEFT` (1)
  - **Right fist runs:** R03, R07, R11, R05, R09, R13 (T2 events) → `RIGHT` (2)
  - **Both feet runs:** R04, R08, R12, R06, R10, R14 (T2 events) → `FORWARD` / `LIFT` (3)
- **Architecture tournament:** MDM, TS+LDA, and FBCSP+LDA compete in 5-fold stratified CV — winner auto-selected
- **Augmentation:** Time-jitter (±25 samples), applied **inside** CV folds only (no data leakage)
- **Artifact rejection:** Epochs with >100µV peak-to-peak discarded (>25% bad channels)
- **Calibration:** Platt scaling on held-out 15% set for well-calibrated probabilities
- **Output:** 4-Class model — saved as `models/moabb_csp_lda.pkl`

---

### Other Classification Engines in the Codebase

ORBIT AI has **three other classifiers** that serve different purposes. They are NOT alternatives that run alongside the Riemannian pipeline — each is used in a specific situation:

#### ERD-Based Classifier (Real Hardware — `bridge_bioamp.py`)

When using the physical BioAmp EXG Pill (which only has 2 channels: C3 and C4), the Riemannian model cannot run because it needs 21 channels. Instead, `bridge_bioamp.py` uses a pure signal-processing approach:

```
C3/C4 raw → Bandpass (1–45 Hz, stateful SOS filter)
  → Welch PSD (Mu+Beta: 8–30 Hz)
  → ERD% = (baseline − live) / baseline
  → 4-way threshold classifier (LIFT / LEFT / RIGHT / IDLE)
  → EMG fast-path: jaw clench detection on every sample (< 100ms)
```

- **No ML model** — pure signal processing with calibrated thresholds
- **LIFT inference:** Bilateral symmetry — C3 and C4 both desynchronize equally
- **LEFT/RIGHT:** Contralateral ERD (left hand imagery → C4 desynchronization, right hand → C3)

#### BioSensor Fallback (`predict_realtime.py` / `predict_websocket.py`)

If no trained MOABB model file (`moabb_csp_lda.pkl`) is found at startup, the dashboard falls back to a simple **Beta/Alpha log-ratio** threshold. This requires no training and provides ~60% accuracy.

#### CNN-LSTM (Legacy — `model.py` + `train.py`)

A deep learning model exists in the codebase but is **not loaded by any live inference script**. It was the original approach before the Riemannian pipeline was adopted.

```
Input (batch, 30, 18) → Conv1D(18→64) → BN → ReLU → MaxPool
  → Conv1D(64→128) → BN → ReLU
  → BiLSTM(128→64, 2 layers) → Global Avg Pool
  → FC(128→64) → FC(64→n_classes)
```

- **Status:** Code exists in `model.py`, trainer in `train.py`, saved weights in `models/best_model.pth` — but **none of the live prediction scripts load it**
- **Why inactive:** It requires subject-specific training data and does not generalize well to new users, which conflicts with the "Universal Brain" design goal
- **See:** [LEGACY.md](./LEGACY.md) for a full list of legacy files and their status

### ⚠️ Known Limitation: Channel Mismatch on Real Hardware

The Riemannian model is trained on 21-channel data, but the BioAmp bridge only provides 2 real channels (C3 and C4), zero-padding the remaining 19. This creates **degenerate covariance matrices** that violate the SPD manifold assumption. When using real hardware through `predict_realtime.py`, the Riemannian model's predictions are unreliable. The ERD classifier in `bridge_bioamp.py` handles this correctly. A future upgrade to **EEGNet** (a channel-agnostic CNN designed for BCI) is planned to unify both paths.

### Algorithm Selection Summary

| Engine | Status | Used When | Channels | Accuracy / Baseline | File(s) |
|--------|--------|-----------|----------|---------------------|---------|
| **Riemannian FBCSP+LDA** | ✅ **Active** | Simulated data or ≥21ch hardware (4-Class) | 21 | 83.71% binary / 4-Class active (25% chance) | `train_moabb.py` → `predict_realtime.py` / `predict_websocket.py` |
| **Model 2 Drift Engine** | ✅ **Active** | Adaptive threshold correction over time | Telemetry | Offsets ±15% | `adapt_drift.py` → `models/drift_profile.json` |
| **ERD Threshold** | ✅ **Active** | Real BioAmp hardware (2 channels) | 2 (C3, C4) | ~70% est. | `bridge_bioamp.py` |
| **BioSensor Fallback** | ⚠️ **Fallback** | No trained model file found | Any | ~60% est. | `predict_realtime.py` / `predict_websocket.py` |
| **CNN-LSTM** | ❌ **Inactive** | Not loaded by any live script | 18 features | Untested | `model.py` + `train.py` |

---

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

For the web dashboards, install Node.js dependencies:
```bash
# Prosthetic Arm Simulator (Recommended)
cd prosthetic-arm-simulator
npm install
cd ..

# Virtual Wheelchair Simulator
cd wheelchair-simulator
npm install
cd ..
```

### 2. The "Universal Brain" Training

```bash
python train_moabb.py
```

This downloads PhysioNet EEG data for 10 subjects, extracts motor imagery and rest epochs, runs an architecture tournament (MDM vs TS+LDA vs FBCSP+LDA), and saves the best model to `models/moabb_csp_lda.pkl`.

### 3. Personal Calibration

IMPORTANT: This requires `simulate_tgam.py` running in a 
separate terminal first, since `calibrate.py` connects to it 
via socket on port 9999.

Terminal 1:
```bash
python simulate_tgam.py
```

Terminal 2 (after Terminal 1 is running):
```bash
python calibrate.py
```

### 4. Real-time Control — Two Interface Options

Choose ONE of the following based on your needs:

#### Option A — Terminal Dashboard (fastest, for quick testing and debugging)

Terminal 1:
```bash
python simulate_tgam.py
```

Terminal 2:
```bash
python predict_realtime.py --demo
```

#### Option B — Full Web Dashboard (recommended for demos and presentations)

Terminal 1:
```bash
python simulate_tgam.py
```

Terminal 2:
```bash
python predict_websocket.py
```

Terminal 3 (Choose your visual simulator):
```bash
# Option 1: Dedicated Prosthetic Arm Simulator (Recommended)
cd prosthetic-arm-simulator
npm run dev
# Open http://localhost:3001

# Option 2: Virtual Wheelchair Simulator
cd wheelchair-simulator
npm run dev
# Open http://localhost:3000
```

#### For Real Hardware (instead of simulate_tgam.py)

Terminal 1:
```bash
python bridge_bioamp.py
```

Terminal 2:
```bash
python predict_realtime.py
# (or python predict_websocket.py for the web dashboard)
```

---

## 🛡️ 5-Layer Safety Pipeline

The `predict_realtime.py` and `predict_websocket.py` engines implement sequential security checks before any prosthetic arm command is issued:

| Gate | Check | Action on Failure |
|------|-------|-------------------|
| **1. Signal Quality** | RMS amplitude > electrode-contact threshold; artifact rejection (±100µV peak-to-peak) | Command blocked |
| **2. Warm-Up** | 5-second "Brain-Settle" period elapsed (0 in demo mode) | Command blocked |
| **3. Fatigue Monitor** | θ/(α+β) ratio below drowsiness limit (from personal profile) | Speed reduced / alert / CRITICAL halt |
| **4. EMG Emergency Stop** | Jaw clench spike > 100μV (via `bridge_bioamp.py`) | Immediate halt (< 100ms) |
| **5. Hysteresis Voting** | 3 consecutive LIFT predictions to engage; 1 IDLE to release (safety-biased) | Prevents jitter/flickering |

---

## 📊 Results

| Metric | Value | Condition |
|--------|-------|-----------|
| **Validation Accuracy** | 83.71% | IDLE vs. LIFT, 21-channel FBCSP+LDA (leakage-free CV + artifact rejection) |
| **Cross-Validation** | 5-Fold Stratified | Best pipeline auto-selected from MDM / TS+LDA / FBCSP+LDA |
| **Inference Latency** | < 100ms | Socket → prediction → display |
| **EMG Stop Latency** | < 100ms | Jaw-clench → STOP command |
| **Calibration Time** | 95 seconds | Personal profiling session |
| **Training Subjects** | 10 | PhysioNet EEGMMIDB clinical EEG subjects |
| **Prototype Cost** | ~₹5,000 (~$60) | BioAmp EXG Pill + ESP32/Arduino |

---

## 📁 Project Structure

```
Mind_Wave_AImodel/
├── config.py                       # Master configuration (ports, thresholds, paths, channel names)
├── requirements.txt                # Python dependencies
│
├── # ── Signal Bridges ────────────────────────────
├── bridge_bioamp.py                # Primary hardware bridge (BioAmp EXG Pill, 2-channel)
├── simulate_tgam.py                # EEG simulator (streams PhysioNet data over socket)
│
├── # ── AI Engine (Active) ────────────────────────
├── universal_model.py              # Universal BCI Model (Model 1) — loads, verifies & runs Universal Brain
├── train_moabb.py                  # Riemannian FBCSP+LDA trainer (primary pipeline)
├── calibrate.py                    # 95-second personal profiler
│
├── # ── AI Engine (Legacy — Not Used at Runtime) ──
├── model.py                        # CNN-LSTM neural network architecture (inactive)
├── train.py                        # CNN-LSTM deep learning trainer (inactive)
├── preprocess.py                   # Feature engineering for CNN-LSTM (inactive)
├── collect_data.py                 # Personal data collection for CNN-LSTM (inactive)
├── fine_tune.py                    # Transfer learning for CNN-LSTM (inactive)
├── evaluate.py                     # Evaluates best_model.pth only (inactive)
├── fetch_and_process_openneuro.py  # OpenNeuro dataset downloader (inactive)
├── quick_process.py                # Welch PSD feature extraction (inactive)
│
├── # ── Runtime ───────────────────────────────────
├── predict_realtime.py             # Live TUI dashboard (uses universal_model.py)
├── predict_websocket.py            # WebSocket server for web dashboard (uses universal_model.py)
├── demo_mode.py                    # One-command demo launcher
│
├── # ── Diagnostics & Reporting ───────────────────
├── diagnose.py                     # System health check + event-aligned testing
├── logger_orbit.py                 # Centralized session logging (CSV with timestamps)
├── auto_report.py                  # Post-session report generator
├── adapt_drift.py                  # Adaptive Neural Drift Engine (Model 2) — adjusts thresholds over time
│
├── # ── Documentation ─────────────────────────────
├── README.md                       # This file
├── ARCHITECTURE.md                 # Architecture deep-dive
├── LEGACY.md                       # Legacy file reference (which files are inactive and why)
├── RND_PAPER.md                    # Full R&D paper with 23 academic references
├── CHANGELOG.md                    # Version history
│
├── # ── Web Dashboards ─────────────────────────────
├── prosthetic-arm-simulator/       # Next.js 15 dedicated robotic arm dashboard (port 3001)
│   ├── src/app/page.tsx            # Interactive SVG articulated arm with spring physics
│   └── ...
├── wheelchair-simulator/           # Next.js virtual wheelchair dashboard (port 3000)
│   ├── src/app/page.tsx            # Main dashboard page
│   └── ...
│
├── data/                           # Training data (.npy, .csv, .edf, cached epochs)
├── models/                         # Saved models & profiles
│   ├── moabb_csp_lda.pkl           # ✅ Active 4-class Riemannian model (~43 MB)
│   ├── drift_profile.json          # 🧬 Adaptive Neural Drift Engine (Model 2) profile
│   ├── personal_profile.json       # Personal calibration data (after running calibrate.py)
│   ├── best_model.pth              # ❌ Legacy CNN-LSTM weights (not loaded at runtime)
│   ├── model1/                     # Universal Brain base model specifications & storage
│   ├── model2/                     # Adaptive Neural Drift engine storage
│   └── prosthetic arm model/       # Specialized prosthetic arm model artifacts
├── logs/                           # Session logs (CSV telemetry)
└── notebooks/                      # Jupyter notebooks
```

---

## 🔮 Project Status & Roadmap

### 📍 Current Stage (v3.1 — Active)
The system is operating at the **4-Class Motor Imagery & Dual-Model BCI System** stage:

- ✅ **4-Class Motor Imagery Decoding**: Upgraded core Riemannian FBCSP+LDA pipeline (`train_moabb.py`) from binary to 4-class classification (`IDLE`, `LEFT`, `RIGHT`, `FORWARD`/`LIFT`) using PhysioNet EEGMMIDB motor execution and imagery runs.
- ✅ **Two-Model Architecture (Model 1 & Model 2)**:
  - **Model 1 (Universal Brain)**: `universal_model.py` + `models/moabb_csp_lda.pkl` — trained once on 10 PhysioNet clinical subjects across 21 motor-cortex channels; strictly read-only at runtime.
  - **Model 2 (Adaptive Neural Drift Engine)**: `adapt_drift.py` + `models/drift_profile.json` — performs longitudinal telemetry analysis (`logs/`), calculating EWMA confidence degradation and linear regression slopes to auto-calibrate threshold offsets without altering base model weights.
- ✅ **Dedicated Prosthetic Arm Simulator**: Fully articulated robotic arm dashboard (`prosthetic-arm-simulator/`, port 3001) built with Next.js 15, Framer Motion spring physics, multi-joint SVG arm rendering, live brain power spectrum, and WebSocket telemetry (`ws://localhost:8765`).
- ✅ **Dual-Simulator Ecosystem**: Complete support for both robotic prosthetic arm simulation (port 3001) and virtual wheelchair simulation (`wheelchair-simulator/`, port 3000).
- ✅ **Modular Model Storage**: Structured directory layout with `models/model1/` (with specifications), `models/model2/`, and `models/prosthetic arm model/`.
- ✅ **5-Layer Medical Safety Pipeline**: Signal Quality RMS Check → 5s Warm-Up Stabilization → Drowsiness/Fatigue Ratio Monitoring → Fast-Path Jaw Clench EMG Emergency Stop (< 100ms) → Multi-Prediction Hysteresis Voting.
- ✅ **95-Second Guided Personal Calibration**: Guided calibration session (`calibrate.py`) for personalizing alpha/beta baselines and fatigue thresholds.
- ✅ **Unified Live Prediction Stack**: Real-time Rich TUI dashboard (`predict_realtime.py`) and WebSocket server (`predict_websocket.py`) using `universal_model.py`.

### 🧬 Adaptive Drift Engine (Model 2)

ORBIT AI now has a **two-model architecture** to handle signal changes from aging, medication, or neurodegeneration:

| Layer | File / Engine | What It Does | Modifies Original? |
|-------|---------------|-------------|-------------------|
| **Model 1** — Universal Brain | `universal_model.py` (`moabb_csp_lda.pkl`) | Core Riemannian classifier, trained once on PhysioNet | **NEVER modified** |
| **Model 2** — Drift Engine | `adapt_drift.py` (`drift_profile.json`) | Adjusts thresholds based on session log analysis | Only writes to its own file |

**How to use:**

```bash
# Analyze your session logs and generate adaptations
python adapt_drift.py

# View drift report without saving
python adapt_drift.py --report

# Reset to original behavior (delete drift profile)
python adapt_drift.py --reset
```

The drift engine reads your accumulated session logs from `logs/`, detects trends (declining confidence, signal amplitude changes, increased fatigue events), and writes threshold adjustments to `models/drift_profile.json`. Both `predict_realtime.py` and `predict_websocket.py` automatically load this file on startup. Deleting `drift_profile.json` instantly reverts to original Model 1 behavior.

### 💡 Innovation Roadmap & Future Ideas

The ORBIT AI ecosystem is actively expanding across medical robotics, assistive mobility, neural decoding, and tele-rehabilitation. For comprehensive technical specifications, refer to [**`future_ideas.md`**](./future_ideas.md).

```
                              ┌───────────────────────────────────┐
                              │      ORBIT AI CORE ECOSYSTEM      │
                              └─────────────────┬─────────────────┘
                                                │
         ┌──────────────────────────────────────┼──────────────────────────────────────┐
         ▼                                      ▼                                      ▼
  🦾 ORBIT-Actuate                       🦽 Smart Wheelchair                    🧠 EEGNet-EXG
  Physical Bionic Arm                    Shared Autonomy Mobility               Channel-Agnostic
  Embedded Micro-ROS                     LiDAR Obstacle Avoidance               Compact Neural Net
```

#### 1. 🦾 Physical Bionic Arm Actuator Bridge (ORBIT-Actuate)
- **Embedded Hardware Interface**: Developing a dedicated serial/CAN-bus and Micro-ROS bridge (`bridge_actuator.py`) to stream 4-class motor predictions directly to an ESP32 controlling 5 independent finger servos and a wrist rotator.
- **Continuous Kinematic Smoothing**: Interpolating discrete motor imagery outputs (`FORWARD/LIFT`, `LEFT`, `RIGHT`) into smooth cubic-spline trajectories, eliminating abrupt mechanical jerks.
- **Hardware-Level EMG Kill-Switch**: Hardwired relay cut-off triggered by jaw-clench EMG spike (> 100μV) in under 20ms, bypassing software queues for fail-safe physical operation.
- *Companion App*: [Prosthetic Arm Simulator](./prosthetic-arm-simulator/README.md) (Port 3001).

#### 2. 🦽 Shared-Autonomy Assistive Mobility Engine
- **Obstacle-Aware Vector Field Navigation**: Fusing decoded mental commands with 2D LiDAR and ultrasonic sensor proximity data using the Vector Field Histogram (VFH+) algorithm.
- **Collaborative Steering**: The user’s neural intent dictates general heading (`FORWARD`, `LEFT`, `RIGHT`), while onboard collision avoidance smoothly steers away from walls, furniture, or stairs.
- **Three.js 3D Digital Twin**: Enhancing the simulation into an interactive 3D hospital/home virtual environment for patient training before physical wheelchair trials.
- *Companion App*: [Wheelchair Simulator](./wheelchair-simulator/README.md) (Port 3000).

#### 3. 🧠 Compact Channel-Agnostic Neural Decoder (EEGNet-EXG)
- **Cross-Density Domain Transfer**: Transitioning from 21-channel clinical caps (PhysioNet) to consumer 1–2 channel bio-potential hardware (BioAmp EXG Pill) via spatial attention and adversarial domain alignment.
- **Ultra-Low Latency Edge Inference**: Sub-3,000 parameter architecture optimized for ONNX Runtime and TensorFlow Lite Micro, achieving < 5ms inference latency on Raspberry Pi 5 or STM32 microcontrollers.
- **Online Riemannian Manifold Tracking**: Real-time geometric covariance recentering on Riemannian manifolds to adapt to electrode impedance drift during live sessions.

#### 4. ⚡ Neuro-Haptic Closed-Loop Sensory Substitution
- **Fingertip Tactile Transduction**: Thin-film piezoresistive pressure sensors mounted on bionic fingertips.
- **Vibrotactile Feedback Matrix**: Linear Resonant Actuator (LRA) collar transmitting grip strength and surface contact to the patient's residual limb via frequency modulation (50–250 Hz).

#### 5. 🏥 Tele-BCI & Longitudinal Clinical Rehabilitation
- **Clinician Dashboard**: Cloud-synced telemetry analyzing daily session logs (`logger_orbit.py`) and automated PDF summaries (`auto_report.py`) to track neuroplastic motor recovery in stroke/ALS rehabilitation.

| Milestone | Target Horizon | Core Deliverables |
| :--- | :---: | :--- |
| **Phase 1: BCI Core & Simulation** | **Current (v3.1)** | 4-Class Riemannian decoding, Dual-Model architecture, Prosthetic & Wheelchair simulators |
| **Phase 2: Embedded Edge & Actuation** | **Q1 2027** | Micro-ROS ESP32 bridge, 3D printed bionic hand actuation, EEGNet-EXG edge export |
| **Phase 3: Shared Autonomy & Haptics** | **Q2 2027** | LiDAR obstacle fusion, vibrotactile sensory wristband, 3D WebGL digital twin |


---

## ⚠️ Troubleshooting

**ImportError: No module named 'logger_orbit'**
Ensure `logger_orbit.py` exists in the project root. Run 
`git pull` to fetch the latest version if missing.

**Connection refused on port 9999**
Start `simulate_tgam.py` FIRST, in its own terminal, before 
running `calibrate.py`, `predict_realtime.py`, or `predict_websocket.py`.

**Web dashboard shows blank page / won't connect**
1. Confirm `predict_websocket.py` is running and shows 
   "WebSocket server started" in the terminal
2. Confirm you ran `npm install` inside `prosthetic-arm-simulator/` 
   or `wheelchair-simulator/` at least once
3. Check the browser console for WebSocket connection errors

**All predictions show IDLE**
Run `python calibrate.py` to set your personal bio-thresholds 
before running `predict_realtime.py` or `predict_websocket.py`.

**ModuleNotFoundError for moabb or pyriemann**
These are required by `train_moabb.py`. Run:
  `pip install moabb pyriemann`

---

## 🔬 Datasets

| Dataset | Subjects | Channels | Rate | Usage in ORBIT AI |
|---------|----------|----------|------|-------------------|
| [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | 109 (10 used) | 64 (21 selected) | 160 Hz | **Active** — Primary training for Riemannian pipeline |
| [OpenNeuro ds002721](https://openneuro.org/datasets/ds002721) | 23 | Variable | Variable | **Legacy** — Was used for CNN-LSTM pre-training |
| [BNCI 2014-001](http://bnci-horizon-2020.eu/database/data-sets#001-2014) | 9 | 22 | 250 Hz | **Reference** — Validation & benchmarking |

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Full system architecture and visual workflows |
| [RND_PAPER.md](./RND_PAPER.md) | Research paper with datasets, algorithms, and 23 academic references |
| [future_ideas.md](./future_ideas.md) | Comprehensive innovation roadmap, robotic actuation & hardware plans |
| [LEGACY.md](./LEGACY.md) | Legacy file reference — which files are inactive and why |
| [CHANGELOG.md](./CHANGELOG.md) | Version history and changes |

---

## ⚠️ Safety Notice

> **ORBIT AI is a research prototype.** It has **not** been certified by any medical device regulatory authority (e.g., FDA, CE). Do not use as the sole control mechanism for a powered prosthetic arm without additional mechanical safety systems (e.g., bump sensors, dead-man switch).

---

## 🤝 Acknowledgements

Built upon the open-source BCI community: [PhysioNet](https://physionet.org) · [MNE-Python](https://mne.tools) · [MOABB](https://moabb.neurotechx.com) · [PyRiemann](https://pyriemann.readthedocs.io) · [Upside Down Labs](https://upsidedownlabs.tech) · [NeuroTechX](https://neurotechx.com)

---

*ORBIT AI v3.1 · September 2026 · NAVEENKCG/Mind_Wave_AImodel*
