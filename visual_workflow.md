# ORBIT AI — Visual Workflow & System Architecture

This document maps how data flows through the entire ORBIT AI codebase — from raw EEG/EMG signals to real-time wheelchair control. It covers offline training, personal calibration, live inference, the 5-layer safety pipeline, and diagnostics. Each diagram annotates the specific scripts and artifacts involved.

---

## 🗺️ 1. Three-Tier System Architecture

```mermaid
flowchart TD
    classDef tier1 fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef tier2 fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef tier3 fill:#020617,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef safety fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff

    subgraph TIER1["Tier 1 — Signal Acquisition"]
        EEG["🧠 EEG Signals<br/>(Scalp Electrodes · Ag/AgCl)"]
        EMG["💪 EMG Signals<br/>(Jaw Electrodes)"]
        BIO["BioAmp EXG Pill<br/>(Gain ~1000× · BW 0.5–50 Hz)"]
        AD["AD8232<br/>(Budget Single-Lead Alt.)"]
        MCU["ESP32 / Arduino<br/>(12-bit ADC → Serial @ 115200)"]

        EEG --> BIO
        EMG --> BIO
        EEG -.-> AD
        BIO --> MCU
        AD -.-> MCU
    end

    subgraph TIER2["Tier 2 — Feature Extraction & Classification"]
        COV["Covariances<br/>(Ledoit-Wolf Shrinkage Estimator)"]
        TS["Tangent Space<br/>Projection at Fréchet Mean"]
        CLASS["Classifier<br/>(LDA / SVM / MDM)<br/>Auto-selected by train_moabb.py"]
        COV --> TS
        TS --> CLASS
    end

    subgraph TIER3["Tier 3 — Safety & Control"]
        GATE["5-Layer Safety Gate<br/>(Signal · Warm-up · Fatigue · EMG · Smoothing)"]:::safety
        TUI["Rich TUI Dashboard<br/>(predict_realtime.py)"]
        CHAIR["Wheelchair Action<br/>(Virtual Arena / Motor)"]
        GATE --> TUI
        GATE --> CHAIR
    end

    MCU -->|"USB Serial<br/>@ 115200 baud"| COV
    CLASS --> GATE

    class EEG,EMG,BIO,AD,MCU tier1
    class COV,TS,CLASS tier2
    class GATE,TUI,CHAIR tier3
```

---

## 🔀 2. Hybrid EEG + EMG Command Mapping

ORBIT AI uses a **"Safety-First" dual-modality** strategy: mental tasks decoded from EEG drive directional commands, while jaw-clench EMG provides a high-speed emergency override channel.

```mermaid
flowchart LR
    classDef eeg fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef emg fill:#450a0a,stroke:#ef4444,stroke-width:2px,color:#fff
    classDef cmd fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef wheel fill:#1e3a5f,stroke:#2196f3,stroke-width:2px,color:#fff

    subgraph EEG_TASKS["Mental Tasks (EEG)"]
        M1("Imagine Moving<br/>Both Feet"):::eeg -->|"Cz Beta ↑"| C1["FORWARD"]:::cmd
        M2("Imagine<br/>Left Hand"):::eeg -->|"C4 α/β shift"| C2["LEFT"]:::cmd
        M3("Imagine<br/>Right Hand"):::eeg -->|"C3 α/β shift"| C3["RIGHT"]:::cmd
        M4("Relaxed,<br/>Eyes Open"):::eeg -->|"α Dominance"| C4["IDLE"]:::cmd
    end

    subgraph EMG_TASKS["Physical Tasks (EMG)"]
        P1("Jaw Clench<br/>/ Bite"):::emg -->|"Spike >100μV"| C5["EMERGENCY<br/>STOP"]:::emg
    end

    WC(("🦽 Wheelchair<br/>Motor")):::wheel

    C1 --> WC
    C2 --> WC
    C3 --> WC
    C4 --> WC
    C5 -.->|"Immediate<br/>Override · <100ms"| WC
```

| Command | Signal | Mental / Physical Task | Bio-Marker | Latency |
|---------|--------|------------------------|------------|---------|
| **FORWARD** | 🧠 EEG | Imagine moving both feet | Cz Central Beta Power ↑ | ~1s window |
| **LEFT** | 🧠 EEG | Imagine squeezing left hand | C4 Alpha/Beta shift | ~1s window |
| **RIGHT** | 🧠 EEG | Imagine squeezing right hand | C3 Alpha/Beta shift | ~1s window |
| **IDLE** | 🧠 EEG | Relaxed state, eyes open | Alpha power dominance | ~1s window |
| **STOP** | 💪 EMG | **Jaw clench (bite teeth)** | High-frequency spike >100μV | **< 100ms** |

---

## 🔧 3. Hardware Connection Path

```mermaid
flowchart LR
    classDef hw fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef afe fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef mcu fill:#0f172a,stroke:#2196f3,stroke-width:2px,color:#fff
    classDef host fill:#020617,stroke:#f59e0b,stroke-width:2px,color:#fff

    subgraph USER["👤 User"]
        SCALP["Scalp Electrodes<br/>(Ag/AgCl)"]:::hw
        JAW["Jaw EMG<br/>Electrodes"]:::hw
    end

    subgraph AFE["Analog Front-End"]
        BIOAMP["BioAmp EXG Pill<br/>Gain ~1000×<br/>BW: 0.5–50 Hz"]:::afe
        AD8232["AD8232<br/>Budget Single-Lead"]:::afe
    end

    subgraph MCU["Microcontroller"]
        ESP["ESP32 / Arduino<br/>ADC → Serial"]:::mcu
    end

    subgraph HOST["🖥️ Host PC"]
        BRIDGE["bridge_bioamp.py<br/>(Serial @ 115200)"]:::host
        BRIDGE_ALT["bridge_ad8232.py<br/>(Bandpass + Ch. Replication)"]:::host
        PIPELINE["ORBIT AI Pipeline<br/>(MNE → PyRiemann → Classifier)"]:::host
        DASHBOARD["Rich TUI Dashboard<br/>(predict_realtime.py)"]:::host
    end

    SCALP -->|EEG| BIOAMP
    JAW -->|EMG| BIOAMP
    SCALP -.->|Alt.| AD8232
    BIOAMP --> ESP
    AD8232 -.-> ESP
    ESP -->|USB Serial| BRIDGE
    ESP -.->|USB Serial| BRIDGE_ALT
    BRIDGE --> PIPELINE
    BRIDGE_ALT -.-> PIPELINE
    PIPELINE --> DASHBOARD
```

---

## 🔄 4. Full Data Pipeline — Offline Training

```mermaid
flowchart TD
    classDef data fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef artifact fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff

    subgraph PHASE1["Phase 1 — Data Sourcing"]
        DS1["PhysioNet EEGMMIDB<br/>109 subjects · 64ch · 160Hz<br/>Motor Imagery + Baseline"]:::data
        DS2["OpenNeuro ds002721<br/>23 subjects · Motor Imagery<br/>BIDS-formatted .edf files"]:::data
        DS3["BNCI 2014-001<br/>9 subjects · 22ch · 250Hz<br/>4-class Motor Imagery"]:::data
        FETCH["fetch_and_process_openneuro.py<br/>Downloads ds002721 + ds003478<br/>Extracts power band features"]:::script
        QP["quick_process.py<br/>Welch PSD → dB-scale<br/>Z-score normalization"]:::script

        DS2 --> FETCH
        FETCH --> QP
        QP --> NPY["data/X_pretrained.npy<br/>shape: (30754, 10, 11)<br/>+ y_pretrained.npy"]:::artifact
    end

    subgraph PHASE2["Phase 2 — Riemannian Pipeline (Primary)"]
        MOABB["train_moabb.py<br/>MOABB framework<br/>Stratified 5-fold CV"]:::script

        DS1 --> MOABB
        DS3 -.-> MOABB

        MOABB -->|"Ensemble Search:<br/>MDM vs LDA vs SVM<br/>Best auto-selected"| BEST["models/universal_brain.pkl<br/>(Best Pipeline Auto-Selected)"]:::artifact
    end

    subgraph PHASE3["Phase 2b — Deep Learning Pipeline (Legacy)"]
        TRAIN["train.py<br/>CNN-LSTM · 50 epochs<br/>AdamW + Cosine Annealing"]:::script
        NPY --> TRAIN
        TRAIN --> PTH["models/best_model.pth<br/>(CNN-LSTM weights)"]:::artifact
    end
```

---

## 🎯 5. Calibration & Personalization

```mermaid
flowchart LR
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef artifact fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef flow fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff

    USER["👤 New User"]:::flow

    subgraph CALIBRATE["calibrate.py — 45-Second Guided Session"]
        REST["10s Eyes-Open<br/>Baseline"]:::script
        FOCUS["10s Active<br/>Focus Task"]:::script
        CLENCH["Jaw Clench<br/>Detection"]:::script
        FATIGUE["Fatigue Threshold<br/>Measurement"]:::script

        REST --> FOCUS --> CLENCH --> FATIGUE
    end

    subgraph COLLECT["collect_data.py — Personal Data Collection"]
        REC["Interactive Recording<br/>of Movement Intentions<br/>via Network Socket"]:::script
    end

    subgraph TUNE["fine_tune.py — Transfer Learning"]
        FT["Freeze Feature Layers<br/>Fine-Tune Classifier<br/>Low Learning Rate"]:::script
    end

    USER --> CALIBRATE
    USER --> COLLECT

    CALIBRATE --> PROF["models/personal_profile.json<br/>α baseline · θ ratio · β reactivity<br/>Blink strength · Fatigue limits"]:::artifact
    COLLECT --> CSV["personal_raw_data.csv"]:::artifact
    CSV --> TUNE
    TUNE --> FINEMODEL["Fine-Tuned Model<br/>(personalized weights)"]:::artifact
```

---

## ⚡ 6. Real-Time Inference Pipeline

```mermaid
flowchart TD
    classDef bridge fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef gate fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef safe fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff
    classDef danger fill:#ef4444,stroke:#333,stroke-width:2px,color:#fff
    classDef output fill:#020617,stroke:#f59e0b,stroke-width:2px,color:#fff

    subgraph BRIDGES["Signal Bridges"]
        B1["bridge_bioamp.py<br/>Reads Serial @ 115200<br/>Primary Hardware"]:::bridge
        B2["bridge_ad8232.py<br/>Budget Alt. + Bandpass<br/>Single→64ch Replication"]:::bridge
        B3["simulate_tgam.py<br/>Streams Medical EDF @ 160Hz<br/>Keyboard-controlled class"]:::bridge
        SOCK{{"Network Socket<br/>127.0.0.1:9999"}}

        B1 --> SOCK
        B2 --> SOCK
        B3 --> SOCK
    end

    subgraph ENGINE["predict_realtime.py — Core Runtime"]
        direction TB
        G1{"1. Signal Quality Gate<br/>RMS > min threshold?<br/>(Electrode contact check)"}:::gate
        G2{"2. Warm-Up Gate<br/>2-min calm period elapsed?<br/>(Brain-settle requirement)"}:::gate
        G3{"3. Fatigue Gate<br/>θ/(α+β) below limit?<br/>(Drowsiness detection)"}:::gate
        G4{"4. EMG Gate<br/>Jaw clench detected?<br/>(>100μV spike)"}:::gate
        G5{"5. Smoothing Gate<br/>Weighted majority voting<br/>(Last 3 predictions)"}:::gate
        AI["Riemannian AI<br/>(Cov → TS → LDA/SVM/MDM)<br/>or CNN-LSTM fallback"]:::safe

        G1 -->|Valid| G2
        G1 -->|"Bad Signal"| BLOCK["🚫 Command Blocked"]:::danger
        G2 -->|Ready| G3
        G2 -->|"Not Ready"| BLOCK
        G3 -->|Safe| G4
        G3 -->|"Drowsy"| SLOW["⚠️ Speed Reduced"]:::danger
        G4 -->|"No Clench"| AI
        G4 -->|"Clench!"| ESTOP["🛑 EMERGENCY STOP"]:::danger
        AI -->|Prediction| G5
    end

    subgraph OUTPUT["Output"]
        ARENA["Virtual Wheelchair Arena<br/>(ASCII Grid + W marker)"]:::output
        LOG["Session Logger<br/>(session_logger.py · 100ms)"]:::output
        ORBIT_LOG["Orbit Logger<br/>(logger_orbit.py · daily CSV)"]:::output
        REPORT["Auto Report<br/>(auto_report.py · PDF)"]:::output
    end

    SOCK -->|"Stream"| G1
    G5 -->|"FORWARD / LEFT / RIGHT / IDLE"| ARENA
    G5 --> LOG
    G5 --> ORBIT_LOG
    LOG --> REPORT
```

---

## 📊 7. Diagnostics & Evaluation

```mermaid
flowchart LR
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef output fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff

    MODEL["Trained Model<br/>(universal_brain.pkl<br/>or best_model.pth)"]

    EVAL["evaluate.py<br/>Classification Report<br/>Confusion Matrix<br/>Permutation Feature Importance"]:::script
    DIAG["diagnose.py<br/>Event-Aligned Testing<br/>System Health Check<br/>GPU/CUDA Status"]:::script
    SESS["session_logger.py<br/>100ms Telemetry<br/>Daily CSV Files"]:::script
    ORBITLOG["logger_orbit.py<br/>Centralized Logging<br/>Timestamps + Commands"]:::script
    AUTO["auto_report.py<br/>Command Histograms<br/>Safety Events<br/>PDF + Text Report"]:::script

    MODEL --> EVAL
    MODEL --> DIAG
    EVAL --> METRICS["Accuracy, F1, Confusion Matrix<br/>Confidence Distributions"]:::output
    DIAG --> HEALTH["System Diagnostics Report<br/>Dependencies · CUDA · Hardware"]:::output
    SESS --> LOGS["data/sessions/*.csv<br/>(timestamp, cmd, confidence,<br/>signal quality, fatigue, θ/β)"]:::output
    ORBITLOG --> DAILYLOGS["Daily CSV Logs<br/>(command history)"]:::output
    LOGS --> AUTO
    AUTO --> PDF["Session Report (PDF)<br/>Histograms + Safety Events"]:::output
```

---

## 📂 8. Complete Project File Map

### Signal Bridges & Simulators

| File | Role | Details |
|------|------|---------|
| `simulate_tgam.py` | Simulator Interface | Streams clinical EEG recordings from `X_pretrained.npy` over socket `127.0.0.1:9999`. Keyboard-controlled class selection (0=IDLE, 1=FORWARD). |
| `bridge_bioamp.py` | Primary Hardware Bridge | Reads raw EEG/EXG from BioAmp EXG Pill via USB serial. Applies 45Hz filter. Broadcasts over socket. |
| `bridge_ad8232.py` | Budget Hardware Bridge | Reads from AD8232 via Arduino. Applies custom bandpass filter. Replicates single input to 64-channel matrix. |

### Data Sourcing & Preprocessing

| File | Role | Details |
|------|------|---------|
| `fetch_and_process_openneuro.py` | Dataset Downloader | Downloads OpenNeuro ds002721 & ds003478. Extracts power band features from Fp1 electrode. |
| `preprocess.py` | Offline Data Pipeline | Loads raw CSV, removes outliers, computes ratio features, sliding windows, robust scaling, data augmentation. |
| `quick_process.py` | Fast Feature Extractor | Welch PSD analysis in dB-scale. Z-score normalization. Outputs sliding windows of shape `(N, 10, 11)`. |

### Brain Engine & Learning

| File | Role | Details |
|------|------|---------|
| `model.py` | Neural Network Definitions | Bidirectional LSTM with Self-Attention + CNN-LSTM hybrid (Conv1D→BiLSTM→FC). |
| `train.py` | Deep Learning Trainer (Legacy) | Binary IDLE/FORWARD training. 50 epochs, AdamW, cosine annealing. |
| `train_moabb.py` | ML Trainer (Primary) | MOABB framework. PhysioNet + BNCI datasets. Auto-selects best pipeline (MDM vs LDA vs SVM). |
| `fine_tune.py` | Transfer Learning | Freezes feature layers, fine-tunes classifier on personal data at low LR. |
| `calibrate.py` | Personal Profiler | 45-second guided session measuring α baseline, θ ratio, β reactivity, blink strength. |
| `collect_data.py` | Data Collector | Interactive recording of movement intentions from live socket stream. |

### Diagnostics, Logging & Reporting

| File | Role | Details |
|------|------|---------|
| `evaluate.py` | Model Evaluator | Classification reports, confusion matrices, confidence distributions, permutation importance. |
| `diagnose.py` | System Diagnostics | Event-aligned offline testing. Reports GPU/CUDA status, dependencies, hardware health. |
| `logger_orbit.py` | Centralized Logger | Records timestamps, commands, confidence, signal quality to daily CSV files. |
| `session_logger.py` | High-Freq Session Logger | 100ms telemetry (command, confidence, signal quality, fatigue, attention, meditation, θ/β). Auto-prints session summary. |
| `auto_report.py` | Report Generator | Post-session PDF/text reports with command histograms and safety event summaries. |

### Real-Time Execution

| File | Role | Details |
|------|------|---------|
| `predict_realtime.py` | Core Runtime + TUI | Listens to streaming bridge, applies 5 safety gates, runs inference, renders virtual wheelchair arena. |

### Configuration & Documentation

| File | Role | Details |
|------|------|---------|
| `config.py` | Master Configuration | COM ports, directories, sample rates, window sizes, safety limits, confidence thresholds, command maps. |
| `requirements.txt` | Dependencies | `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pyserial`, `joblib`, `tqdm`, `rich`, `mne`, `openneuro-py`. |
| `README.md` | Project README | High-level overview, setup guide, and user manual. |
| `ARCHITECTURE.md` | Architecture Deep-Dive | Technical algorithms, features, models, and multi-tier safety layers. |
| `orbit_ai_architecture.md` | Full Architecture Manual | Setup workflows, model layers, configuration guides. |
| `RND_PAPER.md` | R&D Paper | Research paper with datasets, references, algorithms, and results. |
| `FILE_ROLES.md` | File Roles Reference | Comprehensive mapping of each script to its role in the system. |
| `visual_workflow.md` | This Document | Mermaid flowcharts tracking all data pipelines and system architecture. |

---

## 🧠 9. Simplified System Breakdown

| Step | What Happens | Key Scripts | Key Artifacts |
|------|-------------|-------------|---------------|
| **1. Data Sourcing** | Download clinical EEG datasets from PhysioNet & OpenNeuro | `fetch_and_process_openneuro.py`, `quick_process.py` | `X_pretrained.npy`, `y_pretrained.npy` |
| **2. Training** | Build the "Universal Brain" — auto-selects best pipeline via ensemble search | `train_moabb.py` (primary), `train.py` (legacy CNN-LSTM) | `universal_brain.pkl`, `best_model.pth` |
| **3. Calibration** | 45-second personal profiling + optional data collection & fine-tuning | `calibrate.py`, `collect_data.py`, `fine_tune.py` | `personal_profile.json` |
| **4. Signal Bridge** | Connect hardware (BioAmp / AD8232) or run simulator → stream to Port 9999 | `bridge_bioamp.py`, `bridge_ad8232.py`, `simulate_tgam.py` | TCP socket stream |
| **5. Live Inference** | 5-layer safety gate → Riemannian AI → weighted voting → wheelchair command | `predict_realtime.py` | TUI dashboard + wheelchair arena |
| **6. Diagnostics** | Evaluate accuracy, log sessions, generate post-session reports | `evaluate.py`, `diagnose.py`, `session_logger.py`, `logger_orbit.py`, `auto_report.py` | Reports, CSVs, PDFs, confusion matrices |

---

## 🔄 10. Two Operating Modes

### Simulation Mode (Testing & Demos)
```
simulate_tgam.py  →(socket 9999)→  predict_realtime.py --demo
      ↑
 (Keyboard: 0=IDLE, 1=FORWARD)
```
Uses real clinical EEG patterns from PhysioNet — scientifically valid demo without hardware.

### Real Hardware Mode (Live Control)
```
Your Brain → BioAmp EXG Pill → ESP32 → USB Serial → bridge_bioamp.py →(socket 9999)→ predict_realtime.py
```

### Budget Hardware Mode (AD8232)
```
Your Brain → AD8232 → Arduino → USB Serial → bridge_ad8232.py →(socket 9999)→ predict_realtime.py
```

Switch modes by changing `SERIAL_PORT` in `config.py` and removing `--demo`.

---

## 🛡️ 11. Safety Gate Reference

| Gate | Check | Action on Failure | Latency |
|------|-------|--------------------|---------| 
| **1. Signal Quality** | RMS amplitude > electrode-contact threshold | Command blocked | Instant |
| **2. Warm-Up** | 2-minute brain-settle period elapsed | Command blocked | N/A (timer) |
| **3. Fatigue** | θ/(α+β) ratio below drowsiness limit | Speed reduced / alert issued | Continuous monitoring |
| **4. EMG Stop** | Jaw-clench spike > 100μV | Immediate wheelchair halt | **< 100ms** |
| **5. Smoothing** | Weighted majority voting over last 3 predictions | Prevents jitter / flickering | Per-prediction |

---

## 📈 12. Results Summary

| Metric | Value | Condition |
|--------|-------|-----------|
| **Validation Accuracy** | 87%+ | IDLE vs. FORWARD, PhysioNet 5-subject subset |
| **Cross-Validation** | 5-Fold Stratified | Best pipeline auto-selected |
| **Inference Latency** | < 100ms | Socket receive → prediction → display |
| **EMG Stop Latency** | < 100ms | Jaw-clench to STOP command |
| **Calibration Time** | 45 seconds | Personal profiling session |
| **Warm-up Required** | 2 minutes | Brain-settle period before movement |

---

*Updated: June 2026 | ORBIT AI v2.0 — Hybrid BCI-EMG System | NAVEENKCG/Mind_Wave_AImodel*
