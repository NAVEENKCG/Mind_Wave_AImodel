# ORBIT AI — Visual Workflow & System Architecture

This document maps how data flows through the entire ORBIT AI codebase — from raw EEG/EMG signals to real-time wheelchair control. It covers offline training, personal calibration, live inference, and the 5-layer safety pipeline.

---

## 🗺️ 1. Three-Tier System Architecture

```mermaid
flowchart TD
    classDef tier1 fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef tier2 fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef tier3 fill:#020617,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef safety fill:#4caf50,stroke:#333,stroke-width:2px,color:#fff

    subgraph TIER1["Tier 1 — Signal Acquisition"]
        EEG["🧠 EEG Signals<br/>(Scalp Electrodes)"]
        EMG["💪 EMG Signals<br/>(Jaw Electrodes)"]
        BIO["BioAmp EXG Pill<br/>(Gain ~1000×)"]
        AD["AD8232<br/>(Budget Alt.)"]
        MCU["ESP32 / Arduino<br/>(ADC → Serial)"]

        EEG --> BIO
        EMG --> BIO
        EEG -.-> AD
        BIO --> MCU
        AD -.-> MCU
    end

    subgraph TIER2["Tier 2 — Feature Extraction & Classification"]
        COV["Covariances<br/>(LWF Estimator)"]
        TS["Tangent Space<br/>Projection"]
        CLASS["Classifier<br/>(LDA / SVM / MDM)"]
        COV --> TS
        TS --> CLASS
    end

    subgraph TIER3["Tier 3 — Safety & Control"]
        GATE["5-Layer Safety Gate<br/>(Signal, Warm-up, Fatigue, EMG, Smoothing)"]:::safety
        TUI["Rich TUI Dashboard"]
        CHAIR["Wheelchair Action"]
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
    C5 -.->|"Immediate<br/>Override"| WC
```

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
        PIPELINE["ORBIT AI Pipeline<br/>(MNE → PyRiemann → Classifier)"]:::host
        DASHBOARD["Rich TUI Dashboard"]:::host
    end

    SCALP -->|EEG| BIOAMP
    JAW -->|EMG| BIOAMP
    SCALP -.->|Alt.| AD8232
    BIOAMP --> ESP
    AD8232 -.-> ESP
    ESP -->|USB Serial| BRIDGE
    BRIDGE --> PIPELINE
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
        DS1["PhysioNet EEGMMIDB<br/>109 subjects · 64ch · 160Hz"]:::data
        DS2["OpenNeuro ds002721<br/>23 subjects · Motor Imagery"]:::data
        DS3["BNCI 2014-001<br/>9 subjects · 22ch · 250Hz"]:::data
        FETCH["fetch_and_process_openneuro.py"]:::script
        QP["quick_process.py<br/>(Welch PSD → Z-score)"]:::script

        DS2 --> FETCH
        FETCH --> QP
        QP --> NPY["data/X_pretrained.npy<br/>shape: (30754, 10, 11)"]:::artifact
    end

    subgraph PHASE2["Phase 2 — Riemannian Pipeline (Primary)"]
        MOABB["train_moabb.py"]:::script

        DS1 --> MOABB
        DS3 -.-> MOABB

        MOABB -->|"Ensemble Search:<br/>MDM vs LDA vs SVM"| BEST["models/universal_brain.pkl<br/>(Best Pipeline Auto-Selected)"]:::artifact
    end

    subgraph PHASE3["Phase 2b — Deep Learning Pipeline (Legacy)"]
        TRAIN["train.py<br/>(CNN-LSTM · 50 epochs)"]:::script
        NPY --> TRAIN
        TRAIN --> PTH["models/best_model.pth"]:::artifact
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
        REC["Interactive Recording<br/>of Movement Intentions"]:::script
    end

    subgraph TUNE["fine_tune.py — Transfer Learning"]
        FT["Freeze Feature Layers<br/>Fine-Tune Classifier"]:::script
    end

    USER --> CALIBRATE
    USER --> COLLECT

    CALIBRATE --> PROF["models/personal_profile.json<br/>α baseline · θ ratio · β reactivity"]:::artifact
    COLLECT --> CSV["personal_raw_data.csv"]:::artifact
    CSV --> TUNE
    TUNE --> FINEMODEL["Fine-Tuned Model"]:::artifact
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
        B1["bridge_bioamp.py<br/>Reads Serial @ 115200"]:::bridge
        B2["bridge_ad8232.py<br/>Budget Alt. + Bandpass Filter"]:::bridge
        B3["simulate_tgam.py<br/>Streams Medical EDF @ 160Hz"]:::bridge
        SOCK{{"Network Socket<br/>127.0.0.1:9999"}}

        B1 --> SOCK
        B2 --> SOCK
        B3 --> SOCK
    end

    subgraph ENGINE["predict_realtime.py — Core Runtime"]
        direction TB
        G1{"1. Signal Quality Gate<br/>RMS > min threshold?"}:::gate
        G2{"2. Warm-Up Gate<br/>2-min calm period elapsed?"}:::gate
        G3{"3. Fatigue Gate<br/>θ ratio below limit?"}:::gate
        G4{"4. EMG Gate<br/>Jaw clench detected?"}:::gate
        G5{"5. Smoothing Gate<br/>Weighted majority voting"}:::gate
        AI["Riemannian AI<br/>(Cov → TS → LDA/SVM/MDM)"]:::safe

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
        LOG["Session Logger<br/>(100ms telemetry)"]:::output
        REPORT["Auto Report<br/>(Post-Session PDF)"]:::output
    end

    SOCK -->|"Stream"| G1
    G5 -->|"FORWARD / LEFT / RIGHT / IDLE"| ARENA
    G5 --> LOG
    LOG --> REPORT
```

---

## 📊 7. Diagnostics & Evaluation

```mermaid
flowchart LR
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef output fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff

    MODEL["Trained Model<br/>(universal_brain.pkl<br/>or best_model.pth)"]

    EVAL["evaluate.py<br/>Classification Report<br/>Confusion Matrix<br/>Feature Importance"]:::script
    DIAG["diagnose.py<br/>Event-Aligned Testing<br/>System Health Check<br/>GPU/CUDA Status"]:::script
    SESS["session_logger.py<br/>100ms Telemetry<br/>Daily CSV Files"]:::script
    AUTO["auto_report.py<br/>Command Histograms<br/>Safety Events<br/>PDF Report"]:::script

    MODEL --> EVAL
    MODEL --> DIAG
    EVAL --> METRICS["Accuracy, F1, Confusion Matrix"]:::output
    DIAG --> HEALTH["System Diagnostics Report"]:::output
    SESS --> LOGS["data/sessions/*.csv"]:::output
    LOGS --> AUTO
    AUTO --> PDF["Session Report (PDF)"]:::output
```

---

## 🧠 8. Simplified System Breakdown

| Step | What Happens | Key Scripts | Key Artifacts |
|------|-------------|-------------|---------------|
| **1. Data Sourcing** | Download clinical EEG datasets from PhysioNet & OpenNeuro | `fetch_and_process_openneuro.py`, `quick_process.py` | `X_pretrained.npy`, `y_pretrained.npy` |
| **2. Training** | Build the "Universal Brain" — auto-selects best pipeline via ensemble search | `train_moabb.py` (primary), `train.py` (legacy CNN-LSTM) | `universal_brain.pkl`, `best_model.pth` |
| **3. Calibration** | 45-second personal profiling + optional data collection & fine-tuning | `calibrate.py`, `collect_data.py`, `fine_tune.py` | `personal_profile.json` |
| **4. Signal Bridge** | Connect hardware (BioAmp / AD8232) or run simulator → stream to Port 9999 | `bridge_bioamp.py`, `bridge_ad8232.py`, `simulate_tgam.py` | TCP socket stream |
| **5. Live Inference** | 5-layer safety gate → Riemannian AI → weighted voting → wheelchair command | `predict_realtime.py` | TUI dashboard + wheelchair arena |
| **6. Diagnostics** | Evaluate accuracy, log sessions, generate post-session reports | `evaluate.py`, `diagnose.py`, `session_logger.py`, `auto_report.py` | Reports, CSVs, confusion matrices |

---

## 🔄 9. Two Operating Modes

### Simulation Mode (Testing & Demos)
```
simulate_tgam.py  →(socket 9999)→  predict_realtime.py --demo
      ↑
 (Keyboard: 0=IDLE, 1=FORWARD)
```

### Real Hardware Mode (Live Control)
```
Your Brain → BioAmp EXG Pill → ESP32 → USB Serial → bridge_bioamp.py →(socket 9999)→ predict_realtime.py
```

Switch modes by changing `SERIAL_PORT` in `config.py` and removing `--demo`.

---

## 🛡️ 10. Safety Gate Reference

| Gate | Check | Action on Failure |
|------|-------|--------------------|
| **1. Signal Quality** | RMS amplitude > electrode-contact threshold | Command blocked |
| **2. Warm-Up** | 2-minute brain-settle period elapsed | Command blocked |
| **3. Fatigue** | θ/(α+β) ratio below drowsiness limit | Speed reduced / alert |
| **4. EMG Stop** | Jaw-clench spike > 100μV | Immediate wheelchair halt |
| **5. Smoothing** | Weighted majority voting over recent predictions | Prevents jitter / flickering |

---

*Updated: June 2026 | ORBIT AI v2.0 — Hybrid BCI-EMG System | NAVEENKCG/Mind_Wave_AImodel*
