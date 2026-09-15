# 💡 ORBIT AI & Ecosystem — Future Ideas & Innovation Roadmap

This document outlines the high-impact research, product, and hardware ideas extending the **ORBIT AI Brain-Computer Interface** and its surrounding ecosystem.

---

## 📑 Table of Contents
1. [PalmPay — AI-Based Palm Vein Biometric Authentication & Medical Device Gateway](#1-palmpay--ai-based-palm-vein-biometric-authentication--medical-device-gateway)
2. [ORBIT-Actuate — Physical Bionic Arm Hardware & Micro-ROS Bridge](#2-orbit-actuate--physical-bionic-arm-hardware--micro-ros-bridge)
3. [Shared-Autonomy Smart Wheelchair Navigation Engine](#3-shared-autonomy-smart-wheelchair-navigation-engine)
4. [EEGNet-EXG — Channel-Agnostic Compact Neural Decoder](#4-eegnet-exg--channel-agnostic-compact-neural-decoder)
5. [Neuro-Haptic Closed-Loop Sensory Substitution](#5-neuro-haptic-closed-loop-sensory-substitution)
6. [Tele-BCI & Longitudinal Clinical Rehabilitation Platform](#6-tele-bci--longitudinal-clinical-rehabilitation-platform)
7. [Cross-Project Synergy Matrix](#7-cross-project-synergy-matrix)

---

## 1. PalmPay — AI-Based Palm Vein Biometric Authentication & Medical Device Gateway

### 🌟 Executive Overview
**PalmPay** is an AI-powered contactless palm-vein biometric authentication and security platform. Palm vein patterns are subsurface vascular structures that are virtually impossible to forge, replicate, or steal, unlike surface fingerprints or facial imagery.

While designed as a standalone biometric authorization and digital wallet system, PalmPay also serves as the **biometric security gateway for the ORBIT AI BCI medical ecosystem**.

```
                           ┌──────────────────────────────┐
                           │   Near-Infrared (850nm)      │
                           │     Palm Image Capture       │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │   ROI Extraction & Vein      │
                           │   Contrast (CLAHE / Gabor)   │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │  Deep Biometric Embeddings   │
                           │     (ResNet-18 / Siamese)    │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                           ┌──────────────────────────────┐
                           │   Cosine Distance Matching   │
                           │     + Liveness Check         │
                           └──────────────┬───────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
                   ▼                                             ▼
        [Use Case A: PalmPay]                       [Use Case B: ORBIT AI BCI]
     Contactless Payment Auth                      Patient Biometric Lock/Unlock
   Simulated Digital Wallet & Checkout           Load Patient Model 1 & 2 Profiles
   Transaction Auditing & Ledger                 Prevent Unauthorized Arm Operation
```

### 🎯 Key Objectives
- **Subsurface Vascular Mapping**: Utilize 850nm Near-Infrared (NIR) illumination where deoxygenated hemoglobin absorbs NIR light, creating a distinct, high-contrast dark vein pattern.
- **Liveness & Anti-Spoofing**: Enforce multi-spectral reflectance and micro-pulsation analysis to reject high-resolution photos, 3D silicone mockups, or deceased tissue.
- **High-Speed Biometric Verification**: Extract invariant geometric and deep embeddings (< 150ms inference) with False Accept Rate (FAR) < 0.0001% and False Reject Rate (FRR) < 0.01%.
- **Secure Medical Device Pairing**:
  - **BCI Patient Lock**: Unlock motorized wheelchair or robotic arm only when the registered patient presents their palm.
  - **Profile Auto-Switching**: Automatically switch personal calibration parameters (`models/personal_profile.json`) and drift tracking (`models/drift_profile.json`) to the authenticated user.
  - **Caregiver & Clinic Delegation**: Cryptographically sign assistive usage logs for insurance and medical record compliance.

### 🛠️ Architecture & Tech Stack
| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | React 19 + Vite, Tailwind CSS, Lucide Icons | Responsive interactive kiosk UI, live NIR video feed, transaction ledger, and device pairing panel |
| **Backend API** | Python 3.11+, FastAPI, Uvicorn | Async REST & WebSocket server for live template matching and wallet transactions |
| **Vision & AI Core** | OpenCV, PyTorch, Albumentations, NumPy | Gabor wavelets, CLAHE contrast enhancement, Siamese metric learning network (ResNet/MobileNet backbone) |
| **Database & Security** | SQLite / SQLCipher, Argon2id, Fernet / AES-256 | Salting and encryption of raw biometric templates; zero raw image retention |
| **Hardware Target** | Windows + USB IR / Raspberry Pi 5 + NoIR Camera + 850nm LED Ring | Cost-effective edge deployment under ₹4,000 ($48) |

### 🚀 Roadmap & Milestones
- [x] Initial conceptual architecture and vein extraction pipeline design
- [ ] Synthetic vein dataset generation & benchmark baseline
- [ ] NIR camera hardware testing (Raspberry Pi NoIR + 850nm illumination ring)
- [ ] Siamese metric network training on public palm-vein databases (CASIA, PolyU)
- [ ] FastAPI backend + React Vite demo wallet interface
- [ ] Integration with ORBIT AI WebSocket server for biometric device lock/unlock

---

## 2. ORBIT-Actuate — Physical Bionic Arm Hardware & Micro-ROS Bridge

### 🌟 Executive Overview
Currently, ORBIT AI visualizes decoded neural intention on a high-fidelity articulated Next.js 15 SVG simulator (`prosthetic-arm-simulator/`). The next frontier is **actuating physical bionic limbs in the real world**.

```
 🧠 Brain / BioAmp ──► ORBIT AI (universal_model.py) ──► WebSocket / Serial ──► ESP32 Micro-ROS ──► 5× Servo Tendons
                                                                                                    (Physical Bionic Arm)
```

### 🎯 Key Objectives
- **Hardware Bridge**: Create a serial/CAN-bus/Micro-ROS bridge (`bridge_actuator.py`) linking `predict_realtime.py` predictions to an ESP32 microcontroller.
- **Tendon-Driven Bionic Hand**: Control 3D-printed anthropomorphic hands (InMoov / OpenBionics Brunel Hand) with 5 independent finger servos.
- **Continuous Inverse Kinematics (IK)**: Map discrete motor imagery states (`FORWARD/LIFT`, `LEFT`, `RIGHT`, `IDLE`) into continuous trajectory splines with smooth acceleration curves to avoid jerky servo movements.
- **Hardware Emergency Cutoff**: Redundant physical relay triggered by jaw-clench EMG in < 20ms, bypassing software queues.

---

## 3. Shared-Autonomy Smart Wheelchair Navigation Engine

### 🌟 Executive Overview
Moving from a 2D grid simulator (`wheelchair-simulator/`) to a physically viable smart wheelchair requires **Shared Autonomy**: blending imperfect human neural commands with autonomous obstacle avoidance.

```
 [User Motor Imagery] ──► Intended Vector (e.g. FORWARD) ──┐
                                                           ├─► Vector Field Histogram (VFH+) ──► Safe Motor Drive
 [2D LiDAR / Sonar]   ──► Obstacle Proximity Cloud      ──┘
```

### 🎯 Key Objectives
- **Dynamic Window Approach (DWA) & Potential Fields**: Repulsive forces push the wheelchair away from walls and obstacles even if the user issues a `FORWARD` command toward an obstacle.
- **Smooth Intent Blending**: Neural intention sets the general heading; onboard sensors refine velocity and angular steering to ensure smooth, accident-free navigation.
- **Three.js 3D Digital Twin**: Upgrade the web simulator into a 3D environment simulating realistic indoor clinical settings (doors, ramps, obstacles).

---

## 4. EEGNet-EXG — Channel-Agnostic Compact Neural Decoder

### 🌟 Executive Overview
Model 1 (`universal_model.py`) relies on 21 clinical motor-cortex EEG channels from PhysioNet. Low-cost single or dual-channel hardware (e.g., BioAmp EXG Pill) currently requires synthetic spatial zero-padding. **EEGNet-EXG** is an ultra-compact convolutional architecture designed to run directly on arbitrary channel counts.

### 🎯 Key Objectives
- **Depthwise & Separable Convolutions**: Drastically reduce parameter count (< 3,000 parameters) for sub-5ms CPU/MCU inference.
- **Cross-Density Transfer Learning**: Pre-train on high-density clinical data (PhysioNet 64ch), fine-tune with spatial masking on 1–2 channels.
- **Edge Deployment**: Export to ONNX Runtime and TensorFlow Lite Micro for direct execution on microcontrollers or Raspberry Pi.

---

## 5. Neuro-Haptic Closed-Loop Sensory Substitution

### 🌟 Executive Overview
A primary limitation of existing commercial prosthetics is the lack of sensory feedback. Users cannot feel how hard they are gripping an object, leading to dropped items or crushed objects.

```
 Bionic Arm Tactile Sensors ──► Pressure Telemetry ──► Residual Limb Vibrotactile Band ──► Sensory Awareness
```

### 🎯 Key Objectives
- **Fingertip Pressure Sensing**: Integrate FSR (Force-Sensitive Resistors) or piezoresistive sensors at each bionic fingertip.
- **Vibrotactile Feedback Matrix**: Linear Resonant Actuator (LRA) band worn on the user's upper arm or collarbone translating grip pressure into vibration frequency (50–250 Hz).
- **Haptic Simulation in Dashboard**: Render real-time pressure heatmaps and tactile waveforms in the `prosthetic-arm-simulator` dashboard.

---

## 6. Tele-BCI & Longitudinal Clinical Rehabilitation Platform

### 🌟 Executive Overview
Transform ORBIT AI into an end-to-end tele-rehabilitation platform connecting paralyzed patients with occupational therapists and neurologists.

### 🎯 Key Objectives
- **Automated Clinical Reporting**: Expand `auto_report.py` into PDF/FHIR clinical summaries tracking daily motor imagery accuracy, fatigue trends, and neural adaptation rates.
- **Longitudinal Drift Analytics**: Cloud synchronization of `models/drift_profile.json` allowing clinicians to observe neuroplastic recovery in post-stroke patients over months.
- **Remote Calibration Protocol**: Doctor-guided calibration sessions conducted over WebRTC video with real-time EEG telemetry streams.

---

## 7. Cross-Project Synergy Matrix

| Feature / Initiative | ORBIT AI BCI Core | Prosthetic Arm Simulator | Wheelchair Simulator | PalmPay Biometric Gateway |
| :--- | :---: | :---: | :---: | :---: |
| **4-Class Motor Imagery** | Primary Decoder | Articulated Joint Control | Multi-Direction Steering | — |
| **EMG Emergency Stop (<100ms)** | Safety Interlock | Instant Grip Freeze | Instant Dead-Man Brake | — |
| **Model 2 Adaptive Drift** | Telemetry Analysis | Arm Sensitivity Auto-Tune | Joystick Drift Comp. | — |
| **Palm-Vein Authentication** | Patient Lockout | Bionic Arm Pairing | Wheelchair Ignition Lock | Primary Platform |
| **Low-Cost Edge Hardware** | BioAmp EXG (~$15) | 3D Printed Arm (~$45) | Electric Wheelchair Mod | Raspberry Pi + NoIR (~$48) |
| **Open Science (MIT)** | Yes | Yes | Yes | Yes |

---

*Document maintained by the ORBIT AI Open Source Project · Updated September 2026*
