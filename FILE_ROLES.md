# ORBIT AI — File Roles and Responsibilities

This document provides a clear, comprehensive reference mapping each script and file in the ORBIT AI project to its specific role and function in the system architecture.

---

## 🌉 1. Signal Bridges & Simulators (Getting Data In)

- [simulate_tgam.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/simulate_tgam.py): The simulator interface. Accepts any `.edf` file (by URL or local path), extracts multi-channel raw epochs and band-power features, and streams them over local socket `127.0.0.1:9999` to mock a hardware connection for offline testing and demos.
- [bridge_bioamp.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/bridge_bioamp.py): The primary hardware bridge. Reads raw EEG/EXG signals from the BioAmp EXG Pill via a USB serial connection and broadcasts them over local socket.
- [bridge_ad8232.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/bridge_ad8232.py): An alternative budget hardware bridge. Connects to an Arduino reading from an AD8232 sensor, applies custom bandpass filtering, replicates the single input to a 64-channel matrix matching the model architecture, and streams it.

---

## ⚙️ 2. Data Sourcing & Preprocessing (Data Pipeline)

- [fetch_and_process_openneuro.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/fetch_and_process_openneuro.py): Sourcing tool that downloads large clinical EEG datasets (ds002721 & ds003478) from OpenNeuro, extracts power band features from specific electrode channels (like Fp1), and compiles them into pre-training matrices.
- [preprocess.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/preprocess.py): The main offline data processing pipeline. Handles loading raw CSV logs, removing outliers, computing derived ratio features, applying sliding windows, performing robust scaling normalization, and applying data augmentation/undersampling for class balance.
- [quick_process.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/quick_process.py): An accelerated preprocessing script for the ds002721 dataset that uses decibel-scale Welch Power Spectral Density (PSD) analysis to quickly extract normalized training features.

---

## 🧠 3. Brain Engine & Learning (Models & Custom Profiles)

- [model.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/model.py): Defines the neural network architectures in PyTorch, including a Bidirectional LSTM with Self-Attention and a hybrid 1D CNN-LSTM model that captures spatial-temporal patterns in brain waves.
- [train.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/train.py): The core deep learning training script. Loads and filters pre-processed data to binary classes (IDLE vs. FORWARD), balances sample distribution, and trains the CNN-LSTM model with cosine annealing learning rate schedules.
- [train_moabb.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/train_moabb.py): The machine learning model trainer. Uses the MOABB library to fetch medical motor imagery datasets and trains a Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA) pipeline.
- [fine_tune.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/fine_tune.py): Implements transfer learning by loading a pre-trained base model, freezing feature-extraction layers, and fine-tuning weights at a low learning rate on the user's personalized dataset.
- [calibrate.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/calibrate.py): The profiler script. Conducts a quick user test to measure individual baseline resting levels, active focus states, blink strength, and fatigue thresholds. Saves a `personal_profile.json`.
- [collect_data.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/collect_data.py): The personal data collector. Interactively guides the user to record specific movement intentions, compiling data directly from the network socket into `personal_raw_data.csv`.

---

## 📊 4. Diagnostics, Logging & Reporting (Verification & Stats)

- [evaluate.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/evaluate.py): Model evaluation interface. Computes detailed classification reports, saves confusion matrices, confidence distributions, and performs permutation feature importance checks.
- [diagnose.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/diagnose.py): Event-aligned offline pipeline tester. Validates models on raw PhysioNet records and reports system dependencies, GPU/CUDA availability, and hardware status.
- [logger_orbit.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/logger_orbit.py): The centralized real-time logging utility. Automatically records timestamps, predicted commands, confidence, and signal quality metrics to daily CSV files.
- [session_logger.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/session_logger.py): High-frequency session logger. Records telemetry (timestamp, command, confidence, signal quality, fatigue, attention, meditation, theta/beta ratio) every 100ms and auto-prints a detailed session summary on exit.
- [auto_report.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/auto_report.py): Auto-generates post-session reports (text and PDF formats) summarizing command distribution histograms, safety events, and average signal performance.

---

## ⚙️ 5. Real-Time Execution (The Core Engine)

- [predict_realtime.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/predict_realtime.py): The **terminal-based** core runtime dashboard. Listens to the streaming bridge on port 9999, applies a 5-stage safety gate (signal, warmup, fatigue, EMG, smoothing), performs CSP+LDA or BioSensor inference, and renders the 2D virtual wheelchair arena in the terminal using Rich UI.
- [predict_websocket.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/predict_websocket.py): The **web-based** runtime bridge. Applies the same BCI inference pipeline as `predict_realtime.py` but instead of rendering a terminal UI, it broadcasts all state (command, confidence, fatigue, brain power) over a WebSocket server on `ws://localhost:8765` for the Next.js web dashboard to consume.

---

## 🌐 6. Virtual Wheelchair Web Simulator (`wheelchair-simulator/`)

A full-stack **Next.js 15** web application providing a premium visual dashboard for the virtual EEG wheelchair simulation, inspired by the TGAM NeuroSky Mindwave wheelchair project.

### Backend / Server

| File | Role |
| :--- | :--- |
| `package.json` | Node.js project definition and npm scripts (`dev`, `build`, `start`) |
| `next.config.ts` | Next.js 15 configuration (Turbopack enabled) |
| `tsconfig.json` | TypeScript strict mode configuration |
| `postcss.config.mjs` | PostCSS config for Tailwind CSS processing |
| `eslint.config.mjs` | ESLint rules for the Next.js app |

### Application Core (`src/app/`)

| File | Role |
| :--- | :--- |
| `layout.tsx` | Root layout. Loads **Syne** (display) and **DM Sans** (body) via `next/font/google` and applies CSS variable tokens to `<html>`. Defines global SEO metadata. |
| `globals.css` | Design system foundation. Defines all CSS variables (`--bg-base`, `--accent`, `--font-display`, etc.), global resets, animated gradient mesh background, scrollbar styles, and focus rings. |
| `page.tsx` | Main dashboard page. Manages the WebSocket connection to `predict_websocket.py`, wheelchair position state, session counters, and assembles all UI components into the full responsive layout with animated entrance effects. |

### UI Components (`src/components/`)

| File | Role |
| :--- | :--- |
| `Arena.tsx` | The **virtual wheelchair arena**. Renders an animated SVG wheelchair avatar on a grid background that moves upward on `FORWARD` commands, with dynamic glow effects, pulsing ring animation, and border color changes based on command/fatigue state. |
| `StatusPanel.tsx` | Real-time system status card. Shows the current **command** (FORWARD/IDLE), animated **confidence bar**, **signal quality**, **fatigue level** (NORMAL/ALERT/WARNING/CRITICAL), live connection indicator, and session FORWARD/IDLE ratio. |
| `BrainPowerPanel.tsx` | Brain power visualization card. Renders animated spring-driven **Theta**, **Alpha**, and **Beta** power bars with colored fills and percentage values derived from the live EEG stream. |

### Utilities (`src/lib/`)

| File | Role |
| :--- | :--- |
| `animations.ts` | Centralized Framer Motion animation constants (`SPRING_SMOOTH`, `SPRING_SNAPPY`, `EASE_OUT_EXPO`, `fadeInUp`, `staggerContainer`). All components import from here — no inline easing values. |
| `utils.ts` | `cn()` utility function combining `clsx` and `tailwind-merge` for conditional Tailwind class composition. |

---

## 🛠️ 7. Configuration & Documentation

- [config.py](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/config.py): The configuration panel. Defines variables for COM ports, directories, sampling rates, sliding window sizes, safety limits, and classification target command maps.
- [requirements.txt](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/requirements.txt): Declares required Python library dependencies including `torch`, `mne`, `scikit-learn`, `rich`, and `websockets`.
- [README.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/README.md): High-level setup guide and user manual for running the ORBIT AI system.
- [visual_workflow.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/visual_workflow.md): Mermaid-based flowcharts tracking data pipelines, offline training, and live inference.
- [ARCHITECTURE.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/ARCHITECTURE.md): Technical deep-dive of the algorithms, feature extractions, models, multi-tier safety layers, and web simulator architecture.
- [orbit_ai_architecture.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/orbit_ai_architecture.md): The full architectural overview manual, detailing setup workflow diagrams, model layers, and configuration guides.
- [.gitignore](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/.gitignore): Specifies file exclusions for virtual environments, large data files (`.npy`, `.csv`, `.edf`), model checkpoints (`.pth`, `.pkl`), and Next.js build artifacts (`node_modules/`, `.next/`).
- [RND_PAPER.md](file:///c:/Users/Naveenraj/OneDrive/Pictures/Desktop/Documents/Mind_Wave_AImodel/RND_PAPER.md): Research and development reference paper documenting the scientific basis, dataset choices, algorithm rationale, and experimental findings for the ORBIT AI project.
