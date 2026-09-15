# Changelog

All notable changes to the ORBIT AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [v3.1] — 2026-09-14

### Added
- Dedicated Universal Model module (`universal_model.py`) encapsulating Model 1 (Universal Brain) with read-only guarantees, spatial covariance validation, and synthetic benchmark diagnostics
- Model 1 specification files and documentation (`models/model1/README.md`, `models/model1/model_spec.json`)
- 4-class motor imagery decoding pipeline (`IDLE`, `LEFT`, `RIGHT`, `FORWARD`/`LIFT`) in `train_moabb.py`, `config.py`, `predict_realtime.py`, and `predict_websocket.py`
- Dedicated Prosthetic Arm Simulator (`prosthetic-arm-simulator/`, port 3001) with articulated SVG robotic arm, spring physics, and live telemetry
- Modular model storage directory structure (`models/model1/`, `models/model2/`, `models/prosthetic arm model/`)

### Changed
- Refactored `predict_realtime.py` and `predict_websocket.py` to delegate Model 1 loading and inference to `universal_model.py`
- Updated `README.md` and `ARCHITECTURE.md` to reflect `universal_model.py` and Model 1 role
- Refined confidence thresholds and cooldown timings in `config.py` for multi-class classification
- Overhauled and expanded Innovation Roadmap and Future Ideas across all README files (`README.md`, `future_ideas.md`, `prosthetic-arm-simulator/README.md`, `wheelchair-simulator/README.md`, `models/model1/README.md`) covering PalmPay biometric security, physical actuator bridge (ORBIT-Actuate), shared autonomy wheelchair mobility, EEGNet-EXG, and sensory haptics

---

## [v3.0] — 2026-08-26

### Added
- Adaptive Neural Drift Engine (Model 2) implementation (`adapt_drift.py`)
- Dual-model architecture for real-time predictions (`predict_realtime.py`, `predict_websocket.py`)
- Added PalmPay integration ideas to future roadmap (`future_ideas.md`)

### Changed
- Updated architecture documentation to reflect dual-model system (`ARCHITECTURE.md`)
- Enhanced session logger to support drift metrics (`logger_orbit.py`)

---

## [v2.0] — 2025-04-12

### Added
- MOABB universal brain training (`train_moabb.py`)
- Riemannian Geometry processing pipeline (Covariances → Tangent Space → LDA/SVM/MDM)
- Personal calibration system (`calibrate.py`) — 45-second brain fingerprinting
- 5-layer safety pipeline (Signal Quality · Warm-Up · Fatigue · EMG Stop · Voting)
- EMG jaw clench emergency stop (< 100ms response)
- Bridge files for BioAmp EXG Pill (`bridge_bioamp.py`) and AD8232 (`bridge_ad8232.py`)
- Session logging with CSV + JSON summaries (`logger_orbit.py`)
- System health diagnostics (`diagnose.py`)
- One-command demo launcher (`demo_mode.py`)
- `--demo` flag for `predict_realtime.py` (relaxed thresholds, no warmup)
- Auto-generated session reports (`auto_report.py`)
- WebSocket streaming interface (`predict_websocket.py`)

### Changed
- Primary AI engine switched from CNN-LSTM to MOABB CSP+LDA
- Dashboard sidebar now shows brainwave power bars and fatigue state
- Calibration UX improved with retry logic and signal prerequisites check

---

## [v1.0] — 2025-01-01

### Added
- Initial CNN-LSTM model (`model.py` — Conv1D → BiLSTM → FC)
- OpenNeuro dataset processing (`fetch_and_process_openneuro.py`)
- Real-time Rich TUI dashboard (`predict_realtime.py`)
- EEG stream simulator (`simulate_tgam.py`) with clinical data replay
- Domain shift + washout fix for cross-dataset generalisation
- Feature extraction via Welch PSD (`quick_process.py`, `preprocess.py`)
- Live data collection tool (`collect_data.py`)
- Configuration centralised in `config.py`
