# 🧠 ORBIT AI — Universal BCI System (v2.0)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch)
![MOABB](https://img.shields.io/badge/MOABB-1.1-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**ORBIT AI** is a state-of-the-art, open-source **Brain-Computer Interface** designed for medical wheelchair control. It uses a **Hybrid BCI-EMG** architecture that fuses clinical-grade neural decoding (EEG) with high-speed muscle signal detection (EMG) for maximum safety and reliability.

> **83.71% accuracy** on IDLE vs. FORWARD classification using 21-channel FBCSP+LDA pipeline with corrected cross-validation · **< 100ms** emergency stop via jaw-clench EMG detection · **95-second** comprehensive personal calibration

---

## 💡 Product Ideology

### The Problem We Refuse to Ignore

Approximately **5.4 million people** in the United States alone live with some form of paralysis. Worldwide, hundreds of millions of individuals with conditions like ALS, spinal cord injuries, cerebral palsy, and locked-in syndrome are denied the most basic human freedom — **the freedom to move**.

Traditional assistive technologies — joysticks, sip-and-puff controllers, eye-tracking systems — all demand some level of residual physical ability. For those with the most severe impairments, these solutions simply **do not exist**. The mind remains sharp, but the body cannot respond.

**ORBIT AI exists because thought should be enough.**

### Our Vision

> *A world where no conscious mind is trapped in a motionless body — where the mere intention to move is sufficient to move.*

### Our Mission

To build an **open-source, medically safe, and universally accessible** Brain-Computer Interface that translates raw neural signals into reliable wheelchair commands — and to make it available to **everyone**, not just those who can afford proprietary clinical systems costing $50,000+.

### Core Design Principles

| Principle | What It Means in Practice |
| :--- | :--- |
| 🛡️ **Safety Above All** | No wheelchair command is ever issued without passing a 5-layer security pipeline. A single jaw clench halts everything in under 100ms. The system would rather **do nothing** than do the wrong thing. |
| ♿ **Radical Accessibility** | The entire system costs under **₹5,000** (~$60) in hardware. The software is MIT-licensed and runs on consumer laptops. No hospital visit required. No proprietary lock-in. |
| 🧬 **Open Science, Open Code** | Every algorithm, every dataset reference, every training pipeline is public. We publish our research paper ([RND_PAPER.md](./RND_PAPER.md)) alongside our code — because breakthroughs locked behind paywalls don't save lives. |
| 🧠 **The Brain Adapts, So Must We** | No two brains are identical. ORBIT AI trains on a "Universal Brain" model from clinical datasets, then personalizes through a 95-second calibration. The system adapts to **you**, not the other way around. |
| 🤝 **Human Dignity First** | This is not a tech demo. Every design decision — from the fatigue monitor that slows the chair when you're tired, to the warm-up gate that waits until your brain signals stabilize — treats the user as a **person**, not a data source. |
| ⚡ **Hybrid Intelligence** | We don't rely on a single signal. EEG provides the intention. EMG provides the emergency brake. AI provides the pattern recognition. Redundancy is not overhead — it is **survival engineering**. |

### The North Star

ORBIT AI is a stepping stone. Today, we control a virtual wheelchair with two commands. Tomorrow:

- **4-directional navigation** (Forward, Left, Right, Stop) from pure thought
- **Real-world motor integration** via CAN bus or ROS2
- **Edge deployment** on Raspberry Pi / Jetson Nano for portable, offline use
- **Multi-modal fusion** with eye-tracking and head gestures for command richness
- **Clinical trials** with rehabilitation centers and hospitals

We measure success not in accuracy percentages, but in the number of people who regain the ability to **choose where they go**.

> *"The measure of a technology is not what it can do, but who it sets free."*

---

## 📐 Hybrid Command Mapping

We use a **"Safety-First"** dual-modality strategy — mental tasks for direction, jaw clench for instant stop:

| Command | Signal | Mental / Physical Task | Brain Bio-Marker |
| :--- | :--- | :--- | :--- |
| **FORWARD** | 🧠 EEG | Imagine moving both feet | Cz Central Beta Power ↑ |
| **LEFT** | 🧠 EEG | Imagine squeezing left hand | C4 Alpha/Beta shift |
| **RIGHT** | 🧠 EEG | Imagine squeezing right hand | C3 Alpha/Beta shift |
| **IDLE** | 🧠 EEG | Relaxed state, eyes open | Alpha power dominance |
| **STOP** | 💪 EMG | **Jaw clench (bite teeth)** | High-frequency spike >100μV |

---

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+ and npm (only required for the web dashboard)
- For real hardware mode: BioAmp EXG Pill connected via USB, 
  update SERIAL_PORT in config.py

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
| **Signal Processing** | Stateful zero-phase sosfilt (1–45 Hz) + SciPy (Welch PSD) |
| **AI Core — Primary** | MOABB + PyRiemann (Covariances → Tangent Space → LDA/SVM/MDM) |
| **AI Core — Legacy** | PyTorch CNN-LSTM (Conv1D → BiLSTM → FC) |
| **Safety** | 5-Layer Gate Pipeline (Signal Quality · Warm-Up · Fatigue · EMG Stop · Voting) |
| **Interface** | Real-time TUI dashboard with Python `Rich` |
| **Diagnostics** | Session logging (100ms telemetry), auto-generated PDF reports |

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
        GATE --> WC["Wheelchair Action"]
    end

    MCU -->|"USB Serial @ 115200"| COV
    CLS --> GATE

    style GATE fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
```

> 📄 For detailed architecture diagrams and data pipeline flowcharts, see [visual_workflow.md](./visual_workflow.md)
>
> 📄 For the full research paper with references, see [RND_PAPER.md](./RND_PAPER.md)

---

## 🧬 Algorithm & Model Architecture

ORBIT AI uses **three classification engines** — a primary Riemannian pipeline, a legacy CNN-LSTM, and a physics-based fallback. The system auto-selects the best available engine at runtime.

### 1. Riemannian FBCSP + LDA (Primary — `train_moabb.py`)

The deployed model uses **Riemannian geometry** operating on EEG covariance matrices, **not a neural network**. This is the gold standard for motor imagery BCI.

**Pipeline flow:**
```
Raw EEG (21ch × 1s) → Filter Bank (5 sub-bands: δ, θ, α, β, γ)
  → Covariance Matrices (Ledoit-Wolf shrinkage)
  → Riemannian Tangent Space projection
  → Linear Discriminant Analysis (LDA)
  → Platt-calibrated probability
```

**Key details:**
- **Library:** [PyRiemann](https://pyriemann.readthedocs.io) — covariance matrices are treated as points on the SPD (Symmetric Positive Definite) manifold
- **Feature extraction:** Filter-Bank CSP (FBCSP) — splits signal into 5 frequency bands (0.5–4, 4–8, 8–13, 13–30, 30–50 Hz), computes tangent-space features per band, concatenates
- **Architecture tournament:** Three Riemannian architectures compete in 5-fold CV — MDM, TS+LDA, and FBCSP+LDA — the winner is auto-selected
- **Training data:** PhysioNet EEGMMIDB, 10 subjects, 21 motor-cortex channels (sensorimotor strip: FC5–FC6, C5–C6, CP5–CP6)
- **Augmentation:** Time-jitter (±25 samples), applied **inside** CV folds only (leakage-free)
- **Calibration:** Platt scaling on a held-out 15% set for well-calibrated probabilities
- **Output:** Binary — IDLE (0) vs FORWARD (1), 83.71% CV accuracy

### 2. CNN-LSTM Hybrid (Legacy — `model.py`)

A deep learning model exists but is **not used** in the real-time inference pipeline. It was the original approach before the Riemannian pipeline was adopted.

**Architecture:**
```
Input (batch, 30, 18) → Conv1D(18→64, k=3) → BN → ReLU → MaxPool
  → Conv1D(64→128, k=3) → BN → ReLU
  → BiLSTM(128→64, 2 layers) → Global Avg Pool
  → BN → Dropout → FC(128→64) → FC(64→n_classes)
```

**Key details:**
- **Parameters:** ~200K (vs ~2,500 for a comparable EEGNet)
- **Input:** 18 engineered features (8 band powers + attention + meditation + blink + 7 derived ratios) over 30 timesteps
- **Trained by:** `train.py` on personal data collected via `collect_data.py`
- **Status:** Model file exists (`models/best_model.pth`) but `predict_realtime.py` does not load it

### 3. ERD-Based Classification (BioAmp Hardware — `bridge_bioamp.py`)

For 2-channel BioAmp EXG Pill hardware, a signal-processing approach classifies direction using **Event-Related Desynchronization (ERD)**:

```
C3/C4 raw → Bandpass (1–45 Hz, stateful SOS)
  → Welch PSD (Mu+Beta: 8–30 Hz)
  → ERD% = (baseline − live) / baseline
  → 4-way threshold classifier (FORWARD / LEFT / RIGHT / IDLE)
```

**Key details:**
- **No ML model** — pure signal processing with calibrated thresholds
- **EMG fast-path:** Jaw clench detection on every sample (< 100ms latency), bypasses the 1-second epoch accumulation
- **FORWARD inference:** Bilateral symmetry — when C3 and C4 both desynchronize equally (no Cz channel available)

### 4. BioSensor Fallback (`predict_realtime.py`)

If no trained MOABB model is found, falls back to a physics-based **Beta/Alpha ratio** threshold. This is the simplest classifier and requires no training.

### ⚠️ Known Limitation: Channel Mismatch on Real Hardware

The Riemannian model is trained on 21-channel data, but the BioAmp bridge only provides 2 real channels (C3 and C4), zero-padding the remaining 19. This creates **degenerate covariance matrices** that violate the SPD manifold assumption. When using real hardware through `predict_realtime.py`, the Riemannian model's predictions are unreliable. The ERD classifier in `bridge_bioamp.py` handles this correctly. A future upgrade to **EEGNet** (a channel-agnostic CNN designed for BCI) is planned to unify both paths.

### Algorithm Selection Summary

| Engine | Used When | Channels | Accuracy | File |
|--------|-----------|----------|----------|------|
| **Riemannian FBCSP+LDA** | MOABB model exists + ≥21 channels | 21 | 83.71% | `train_moabb.py` |
| **ERD Threshold** | Real BioAmp hardware | 2 (C3, C4) | ~70% (estimated) | `bridge_bioamp.py` |
| **BioSensor Fallback** | No trained model available | Any | ~60% | `predict_realtime.py` |
| **CNN-LSTM** | Not currently active | 18 features | Untested | `model.py` + `train.py` |

---

## 🚀 Getting Started

### 1. Installation

```bash
pip install -r requirements.txt
```

For the web dashboard, also install Node.js dependencies:
```bash
cd wheelchair-simulator
npm install
cd ..
```

### 2. The "Universal Brain" Training

```bash
python train_moabb.py
```

### 3. Personal Calibration

IMPORTANT: This requires simulate_tgam.py running in a 
separate terminal first (see Step 4a below), since 
calibrate.py connects to it via socket on port 9999.

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

Terminal 3:
```bash
cd wheelchair-simulator
npm run dev
```

Then open http://localhost:3000 in your browser to see the 
live wheelchair arena, brain power panel, and status dashboard.

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

The `predict_realtime.py` engine implements sequential security checks before any wheelchair command is issued:

| Gate | Check | Action on Failure |
|------|-------|-------------------|
| **1. Signal Quality** | RMS amplitude > electrode-contact threshold | Command blocked |
| **2. Warm-Up** | 2-minute "Brain-Settle" period elapsed | Command blocked |
| **3. Fatigue Monitor** | θ/(α+β) ratio below drowsiness limit | Speed reduced / alert |
| **4. EMG Emergency Stop** | Jaw clench spike > 100μV | Immediate halt (< 100ms) |
| **5. Weighted Voting** | Majority vote over last 3 predictions | Prevents jitter/flickering |

---

## 📊 Results

### Current System Results
| Metric | Value | Condition |
|--------|-------|-----------|
| **Validation Accuracy** | 83.71% | IDLE vs. FORWARD, 21-channel FBCSP+LDA (data leakage resolved & artifact rejection) |
| **Cross-Validation** | 5-Fold Stratified | Best pipeline auto-selected |
| **Inference Latency** | < 100ms | Socket → prediction → display |
| **EMG Stop Latency** | < 100ms | Jaw-clench → STOP command |
| **Calibration Time** | 95 seconds | Personal profiling session (includes L/R imagery) |

### Overall Project Benchmarks
| Metric | Value |
|--------|-------|
| Model | MOABB FBCSP+LDA |
| Cross-validation Accuracy | 83.71% (PhysioNet Motor Imagery) |
| Inference Latency | < 1 second |
| Training Subjects | 9+ clinical EEG subjects |
| Emergency Stop Response | < 100ms (jaw clench EMG) |
| Prototype Cost | ~₹5,000 |

---

## 📁 File Structure

orbit-ai/
├── predict_realtime.py   ← Main dashboard
├── simulate_tgam.py      ← EEG stream simulator
├── train_moabb.py        ← Universal brain training
├── calibrate.py          ← Personal calibration
├── diagnose.py           ← System health check
├── bridge_bioamp.py      ← Real hardware bridge
├── bridge_ad8232.py      ← AD8232 EMG bridge
├── logger_orbit.py       ← Session logging
├── config.py             ← All settings
├── model.py              ← CNN-LSTM architecture
└── models/
    ├── moabb_csp_lda.pkl ← Trained MOABB model
    └── personal_profile.json ← Your calibration

---

## 📁 Project Structure

```
Mind_Wave_AImodel/
├── config.py                       # Master configuration (ports, thresholds, paths)
├── requirements.txt                # Python dependencies
│
├── # ── Signal Bridges ────────────────────────────
├── bridge_bioamp.py                # Primary hardware bridge (BioAmp EXG Pill)
├── bridge_ad8232.py                # Budget hardware bridge (AD8232)
├── simulate_tgam.py                # EEG simulator (streams clinical data over socket)
│
├── # ── Data Pipeline ─────────────────────────────
├── fetch_and_process_openneuro.py  # Downloads OpenNeuro datasets
├── quick_process.py                # Welch PSD feature extraction
├── preprocess.py                   # Full offline preprocessing pipeline
│
├── # ── AI Engine ─────────────────────────────────
├── model.py                        # CNN-LSTM neural network architecture
├── train.py                        # Deep learning trainer (legacy)
├── train_moabb.py                  # Riemannian pipeline trainer (primary)
├── calibrate.py                    # 95-second personal profiler
├── collect_data.py                 # Personal data collection
├── fine_tune.py                    # Transfer learning fine-tuning
│
├── # ── Runtime ───────────────────────────────────
├── predict_realtime.py             # Core runtime + TUI dashboard
│
├── # ── Diagnostics & Reporting ───────────────────
├── evaluate.py                     # Model evaluation (accuracy, F1, confusion matrix)
├── diagnose.py                     # System health check + event-aligned testing
├── logger_orbit.py                 # Centralized daily logging
├── auto_report.py                  # Post-session PDF/text report generator
│
├── # ── Documentation ─────────────────────────────
├── README.md                       # This file
├── ARCHITECTURE.md                 # Architecture deep-dive
├── orbit_ai_architecture.md        # Full architecture manual
├── visual_workflow.md              # Mermaid flowcharts & data pipeline diagrams
├── FILE_ROLES.md                   # File-by-file role reference
├── RND_PAPER.md                    # Full R&D paper with references
│
├── data/                           # Training data (.npy, .csv, .edf)
├── models/                         # Saved models (.pkl, .pth, .json)
└── notebooks/                      # Jupyter notebooks
```

---

## ⚠️ Troubleshooting

**ImportError: No module named 'logger_orbit'**
Ensure logger_orbit.py exists in the project root. Run 
git pull to fetch the latest version if missing.

**Connection refused on port 9999**
Start simulate_tgam.py FIRST, in its own terminal, before 
running calibrate.py, predict_realtime.py, or predict_websocket.py.

**Web dashboard shows blank page / won't connect**
1. Confirm predict_websocket.py is running and shows 
   "WebSocket server started" in the terminal
2. Confirm you ran npm install inside wheelchair-simulator/ 
   at least once
3. Check the browser console for WebSocket connection errors

**All predictions show IDLE**
Run python calibrate.py to set your personal bio-thresholds 
before running predict_realtime.py or predict_websocket.py.

**ModuleNotFoundError for moabb or pyriemann**
These are required by train_moabb.py. Run:
  pip install moabb pyriemann

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | Full system architecture, file roles, and visual workflows |
| [RND_PAPER.md](./RND_PAPER.md) | Research paper with datasets, algorithms, and 23 academic references |

---

## 🔬 Datasets

| Dataset | Subjects | Channels | Rate | Usage |
|---------|----------|----------|------|-------|
| [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | 109 | 64 | 160 Hz | Primary training (Riemannian pipeline) |
| [OpenNeuro ds002721](https://openneuro.org/datasets/ds002721) | 23 | Variable | Variable | Legacy CNN-LSTM pre-training |
| [BNCI 2014-001](http://bnci-horizon-2020.eu/database/data-sets#001-2014) | 9 | 22 | 250 Hz | Validation & benchmarking |

---

## ⚠️ Safety Notice

> **ORBIT AI is a research prototype.** It has **not** been certified by any medical device regulatory authority (e.g., FDA, CE). Do not use as the sole control mechanism for a powered wheelchair without additional mechanical safety systems (e.g., bump sensors, dead-man switch).

---

## 🤝 Acknowledgements

Built upon the open-source BCI community: [PhysioNet](https://physionet.org) · [MNE-Python](https://mne.tools) · [MOABB](https://moabb.neurotechx.com) · [PyRiemann](https://pyriemann.readthedocs.io) · [Upside Down Labs](https://upsidedownlabs.tech) · [NeuroTechX](https://neurotechx.com)

---

*ORBIT AI v2.0 · June 2026 · NAVEENKCG/Mind_Wave_AImodel*
