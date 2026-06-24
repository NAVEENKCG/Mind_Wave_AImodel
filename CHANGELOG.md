# Changelog

All notable changes to the ORBIT AI project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
