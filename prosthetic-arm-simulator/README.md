# 🦾 ORBIT AI — Prosthetic Arm BCI Simulator

An interactive, high-fidelity robotic arm simulation dashboard for the **ORBIT AI Brain-Computer Interface**. Built with Next.js 15 and Framer Motion, it visualizes real-time EEG-decoded motor imagery commands as articulated bionic arm kinematics, backed by sub-100ms EMG emergency stop protection.

---

## 🌟 Overview

The **Prosthetic Arm Simulator** (running on port **3001**) is the primary robotic neuroprosthetic interface of the ORBIT AI ecosystem:

- **Kinematic Articulated Arm**: Interactive multi-segment SVG rendering featuring shoulder elevation, elbow flexion, wrist rotation, and 3-finger gripper actuation with Framer Motion spring physics.
- **Dual-Model Inference Telemetry**: Receives live predictions from Model 1 (`universal_model.py`) and threshold adaptations from Model 2 (`adapt_drift.py`) over WebSocket (`ws://localhost:8765`).
- **5-Layer Medical Safety Display**: Visualizes signal RMS quality gates, warm-up stabilization states, fatigue indices, and instant jaw-clench EMG braking.
- **Real-Time Band Power Spectrum**: Live frequency distribution tracking Theta (drowsiness), Alpha (relaxation), and Beta (active motor imagery).

---

## 🕹️ Command & Kinematic Mapping

| Command | Modality | Mental / Physical Task | Prosthetic Arm Movement |
| :--- | :---: | :--- | :--- |
| **`LIFT / FORWARD`** | 🧠 EEG | Imagine moving both feet | Elevates shoulder joint + flexes elbow upward |
| **`LEFT`** | 🧠 EEG | Imagine squeezing left hand | Rotates wrist counter-clockwise / opens gripper |
| **`RIGHT`** | 🧠 EEG | Imagine squeezing right hand | Rotates wrist clockwise / closes gripper |
| **`IDLE`** | 🧠 EEG | Relaxed baseline, open eyes | Holds current position in resting equilibrium |
| **`STOP`** | 💪 EMG | **Jaw clench (bite teeth)** | **EMG Safety Freeze** — halts motion in < 100ms |

---

## 🛠️ Architecture & Tech Stack

| Layer | Technology | Details |
| :--- | :--- | :--- |
| **Framework** | Next.js 15 (App Router), React 19, TypeScript | Strict TypeScript mode, server/client component separation |
| **Styling** | Tailwind CSS (CSS Variables) | Dark-first medical aesthetic (`#050A18`, `#0D1526`, `#152035`) |
| **Physics & Motion** | Framer Motion | Spring physics (`stiffness: 300, damping: 30`), layout transitions |
| **Icons** | Lucide React | High-contrast UI iconography |
| **Networking** | Native WebSocket Client | Connects to `ws://localhost:8765` with automated retry logic |

### Core UI Modules
- **`ArmArena.tsx`**: Articulated SVG robotic arm with reactive spring physics, joint rotation nodes, and dynamic posture transitions.
- **`StatusPanel.tsx`**: Connection status, model prediction confidence meter, signal RMS health, and fatigue monitor.
- **`BrainPowerPanel.tsx`**: Spectral bar charts visualizing Theta, Alpha, and Beta wave power in real time.

---

## ⚡ Quick Start

### 1. Start the BCI Inference Engine
In your project root directory:

```bash
# Terminal 1: Start synthetic EEG stream (or connect real BioAmp hardware)
python simulate_tgam.py

# Terminal 2: Start the WebSocket server
python predict_websocket.py --demo
```

### 2. Launch the Prosthetic Arm Simulator
In this directory (`prosthetic-arm-simulator/`):

```bash
npm install
npm run dev
```

Open [http://localhost:3001](http://localhost:3001) in your browser.

---

## 💡 Future Ideas & Engineering Roadmap

```
 🧠 Decoded EEG ──► Inverse Kinematics (IK) ──► WebSocket / Web Serial ──► ESP32 Micro-ROS ──► 5× Servo Tendons
                                                                                              (Physical Arm)
```

### 1. 🦾 Physical Actuator Bridge (Micro-ROS & CAN-Bus)
- Connect this web dashboard directly to real physical bionic limbs (e.g. InMoov or OpenBionics Brunel Hand).
- Implement a **Web Serial API** bridge in the browser to stream servo angles directly to an ESP32 microcontroller over USB, or via Micro-ROS over WiFi/Ethernet.

### 2. 📐 Multi-DOF Inverse Kinematics (IK) Solver
- Upgrade the 2D SVG kinematic chain to a full 6-DOF Inverse Kinematics solver.
- Allow users to execute complex reach-and-grasp tasks with automated collision-free end-effector path planning.

### 3. 🌐 3D WebGL / Three.js Digital Twin
- Render an anatomically precise 3D prosthetic arm model using Three.js / React Three Fiber.
- Support photorealistic textures, muscle tendon rigging, and physical object interaction (picking up cups, turning keys).

### 4. ⚡ Closed-Loop Sensory Feedback & Haptic Heatmaps
- Integrate thin-film pressure sensors at the virtual and physical fingertips.
- Render dynamic tactile pressure heatmaps in `ArmArena` while triggering wireless vibrotactile feedback bands worn by the amputee.

### 5. 🔐 PalmPay Biometric Grip & Device Authorization
- Cross-integrate with [**PalmPay**](../future_ideas.md#1-palmpay--ai-based-palm-vein-biometric-authentication--medical-device-gateway).
- Patient must present their palm to unlock bionic arm movement, protecting against accidental activation or tampering when the prosthesis is detached.

---

## 🔗 Related Documentation
- [ORBIT AI Main Documentation](../README.md)
- [Future Ideas & Comprehensive Roadmap](../future_ideas.md)
- [Wheelchair Simulator](../wheelchair-simulator/README.md)
- [System Architecture](../ARCHITECTURE.md)
