# System Architecture

---

##  High-Level Architecture

```mermaid
flowchart TD
    U[" User"] --> UI[" Web Interface"]
    UI --> API[" FastAPI Backend"]
    API --> DA[" Detection Agent"]
    DA --> IR[" Input Router"]

    IR --> TD[" Text Detector"]
    IR --> ID[" Image Detector"]
    IR --> AD[" Audio Detector"]
    IR --> VD[" Video Detector"]

    TD --> EA[" Evidence Aggregator"]
    ID --> EA
    AD --> EA
    VD --> EA

    EA --> FR[" Final Report"]

    style U fill:#e3f2fd,stroke:#1976d2,color:#111827,stroke-width:2px
    style UI fill:#e8f5e9,stroke:#2e7d32,color:#111827,stroke-width:2px
    style API fill:#fff3e0,stroke:#ef6c00,color:#111827,stroke-width:2px
    style DA fill:#f3e5f5,stroke:#7b1fa2,color:#111827,stroke-width:2px
    style IR fill:#fce4ec,stroke:#c2185b,color:#111827,stroke-width:2px
    style EA fill:#e0f7fa,stroke:#00838f,color:#111827,stroke-width:2px
    style FR fill:#fff8e1,stroke:#f9a825,color:#111827,stroke-width:3px
```

---

##  Processing Pipeline

The system follows a modular processing pipeline:

```text
  USER
   │
   ▼
  WEB INTERFACE
   │
   ▼
  FASTAPI BACKEND
   │
   ▼
  DETECTION AGENT
   │
   ▼
  INPUT ROUTER
   │
   ├──────────────┬──────────────┬──────────────┐
   ▼              ▼              ▼              ▼
  TEXT          IMAGE           AUDIO            VIDEO
DETECTOR       DETECTOR       DETECTOR       DETECTOR
   │              │              │              │
   └──────────────┴──────────────┴──────────────┘
                          │
                          ▼
                      EVIDENCE
                   AGGREGATOR
                          │
                          ▼
                      FINAL REPORT
```

---

#  Complete System Architecture

```mermaid
flowchart TD
    USER[" User"]
    WEB[" Web Interface"]
    API[" FastAPI Backend"]
    AGENT[" Detection Agent"]
    ROUTER[" Input Router"]

    TEXT[" Text Detector"]
    IMAGE[" Image Detector"]
    AUDIO[" Audio Detector"]
    VIDEO[" Video Detector"]

    FRAMES[" Frame Extraction"]
    AUDIO_EXT[" Audio Extraction"]

    AGG[" Evidence Aggregator"]
    REPORT[" Final Report"]

    USER --> WEB
    WEB --> API
    API --> AGENT
    AGENT --> ROUTER

    ROUTER --> TEXT
    ROUTER --> IMAGE
    ROUTER --> AUDIO
    ROUTER --> VIDEO

    VIDEO --> FRAMES
    VIDEO --> AUDIO_EXT

    FRAMES --> IMAGE
    AUDIO_EXT --> AUDIO

    TEXT --> AGG
    IMAGE --> AGG
    AUDIO --> AGG
    AGG --> REPORT

    style USER fill:#e3f2fd,stroke:#1565c0,color:#111827,stroke-width:2px
    style WEB fill:#e8f5e9,stroke:#2e7d32,color:#111827,stroke-width:2px
    style API fill:#fff3e0,stroke:#ef6c00,color:#111827,stroke-width:2px
    style AGENT fill:#f3e5f5,stroke:#7b1fa2,color:#111827,stroke-width:2px
    style ROUTER fill:#fce4ec,stroke:#c2185b,color:#111827,stroke-width:2px
    style AGG fill:#e0f7fa,stroke:#00838f,color:#111827,stroke-width:3px
    style REPORT fill:#fff8e1,stroke:#f9a825,color:#111827,stroke-width:3px
```

---

#  Architecture Principles

| Principle             | Description                                         |
| --------------------- | --------------------------------------------------- |
|  **Modular**        | Each modality has its own specialized detector      |
|  **Extensible**     | New detectors can be added through the router       |
|  **Scalable**        | FastAPI provides a lightweight asynchronous backend |
|  **Intelligent**    | Detection Agent orchestrates the analysis pipeline  |
|  **Evidence-Based** | Results are supported by detector-level evidence    |
|  **Multi-Modal**    | Video combines visual and audio analysis            |
|  **Unified Output** | All evidence is consolidated into one final report  |

---
