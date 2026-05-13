# ORBIT AI — Visual Workflow & Step-by-Step Simulation

This document maps how data moves through the codebase and provides a straightforward guide to running a simulation.

## 🗺️ Visual Architecture

```mermaid
graph TD
    %% Styling Classes
    classDef hardware fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef model fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef ui fill:#020617,stroke:#f59e0b,stroke-width:2px,color:#fff

    subgraph Step 1: Input Sources
        BIO[Real Hardware: BioAmp Pill]:::hardware
        SIM[Simulator: EDF Medical File]:::hardware
    end

    subgraph Step 2: Signal Bridges
        B_BIO(bridge_bioamp.py<br/>Reads serial data)
        B_SIM(simulate_tgam.py<br/>Reads EDF links)
        TCP((Network Socket<br/>Port 9999))
        
        BIO --> B_BIO
        SIM --> B_SIM
        B_BIO -->|Raw Signals| TCP
        B_SIM -->|Raw Signals| TCP
    end

    subgraph Step 3: Intelligence & Inference
        TRAIN[train_moabb.py]:::script
        CAL[calibrate.py]:::script
        PREDICT[predict_realtime.py<br/>AI Core & Safety Gates]:::script
        MODEL[moabb_csp_lda.pkl]:::model
        PROF[personal_profile.json]:::model
        
        TRAIN -->|Creates| MODEL
        CAL -->|Creates| PROF
        
        TCP -->|Streams to| PREDICT
        MODEL -.->|Used by| PREDICT
        PROF -.->|Used by| PREDICT
    end

    subgraph Step 4: Output / Execution
        ARENA(Virtual Wheelchair Arena):::ui
        PREDICT -->|FORWARD / IDLE| ARENA
    end

    %% Class Attachments
    class BIO,SIM hardware
    class B_BIO,B_SIM,TRAIN,CAL,PREDICT script
    class MODEL,PROF model
    class ARENA ui
```

## 🧠 System Breakdown (Simplified)

1. **Input:** You either connect the actual **BioAmp hardware** or run the **Simulator** using a medical dataset link.
2. **Network Bridge:** This data is forwarded locally to Port 9999 so the AI can listen to it.
3. **Intelligence Setup:** `train_moabb.py` builds the core AI brain, and `calibrate.py` tunes it to your resting state.
4. **The Live Core:** `predict_realtime.py` listens to Port 9999, checks for signal noise/fatigue (safety gates), and asks the AI for a decision.
5. **Execution:** The final command (e.g., `FORWARD`) moves the virtual wheelchair in the dashboard.

