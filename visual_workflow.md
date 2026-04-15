# ORBIT AI — Visual Workflow Architecture

This flowchart visually maps how data moves through the codebase, from your real-world hardware into the mathematical core of the AI, and lastly into the Virtual Arena.

```mermaid
graph TD
    %% Styling Classes
    classDef hardware fill:#1e293b,stroke:#0ea5e9,stroke-width:2px,color:#fff
    classDef script fill:#334155,stroke:#a855f7,stroke-width:2px,color:#fff
    classDef model fill:#0f172a,stroke:#22c55e,stroke-width:2px,color:#fff
    classDef ui fill:#020617,stroke:#f59e0b,stroke-width:2px,color:#fff

    subgraph Data Sources [1. Input / Hardware]
        BIO[BioAmp EXG Pill]:::hardware
        SIM[PhysioNet Medical Datasets]:::hardware
    end

    subgraph Networking [2. Signal Bridges]
        B_BIO(bridge_bioamp.py)
        B_SIM(simulate_tgam.py)
        TCP((TCP Socket<br/>Port 9999))
        
        BIO -->|Analog Serial| B_BIO
        SIM -->|EDF Arrays| B_SIM
        B_BIO -->|JSON Stream| TCP
        B_SIM -->|JSON Stream| TCP
    end

    subgraph Training & Brain Profiling [3. Intelligence]
        TRAIN[train_moabb.py]:::script
        CAL[calibrate.py]:::script
        MODEL[models/moabb_csp_lda.pkl]:::model
        PROF[models/personal_profile.json]:::model
        
        TRAIN -->|Saves Pre-Trained| MODEL
        TCP -->|User Brainwaves| CAL
        CAL -->|Saves Bio-Baselines| PROF
    end

    subgraph Inference Core [4. predict_realtime.py]
        direction TB
        GATE{Signal Gate<br/>Noisy / Flat?}
        FATIGUE{Fatigue Gate<br/>Theta Ratio > 1.5?}
        PREDICT[MOABB Pipeline<br/>Inference]
        SMOOTH[Weighted Vote<br/>Debounce]
        
        TCP --> GATE
        MODEL -.-> PREDICT
        PROF -.-> FATIGUE
        
        GATE --Valid--> FATIGUE
        FATIGUE --Safe--> PREDICT
        FATIGUE --Critical--> STOP[Force IDLE]
        PREDICT --> SMOOTH
    end

    subgraph Output [5. Execution]
        ARENA(Virtual Wheelchair Arena):::ui
        WEB(Next.js Web Dashboard):::ui
        
        SMOOTH -->|Move Command| ARENA
        SMOOTH -.->|WebSocket / REST| WEB
        STOP --> ARENA
    end

    %% Class Attachments
    class BIO,SIM hardware
    class B_BIO,B_SIM,TRAIN,CAL script
    class MODEL,PROF model
    class ARENA,WEB ui
```

## System Breakdown

1. **Input:** Data comes from either real-world hardware (`bioamp`) or a software simulator (`tgam`).
2. **Network Bridge:** This data is converted to JSON and fired continuously via a local networking socket (so multiple scripts can read it simultaneously).
3. **Intelligence Setup:** The heavy math takes place here. `train_moabb` creates the general AI brain, while `calibrate.py` creates a "profile" of your normal resting states.
4. **The Live Core:** `predict_realtime.py` acts as the security guard. It ensures the signal is good, ensures you aren't falling asleep, asks the AI for a decision, and smooths the result.
5. **Execution:** The final command moves the virtual machine!
