# Multimodal AI Content Detection Agent

**Status:** Day 7 : Three-Model Text Detection Ensemble

## Description
A multimodal system designed to detect potentially AI-generated text, images, audio, and video using open-source pretrained models.

## Project Objective
Develop a resource-efficient AI detection system leveraging pretrained models without fine-tuning in Version 1.

## Modalities Progress
* 🟢 **Text:** Three-model detector and majority-vote agent implemented
* ⚪ **Image:** Planned
* ⚪ **Audio:** Planned
* ⚪ **Video:** Planned

## Tech Stack
* **Core Language & ML Frameworks:** Python, PyTorch, Transformers
* **AI & Agent Workflows:** Hugging Face, LangGraph
* **Backend & Interface:** FastAPI, Gradio
* **Testing & Quality:** Pytest
* **Version Control & Compute:** GitHub Codespaces, Google Colab

## Architecture & Model Decisions

### Text Detection Architecture (V1)
* **Primary Detector (English):** `Oxidane/tmr-ai-text-detector` (RoBERTa-base fine-tuned on 50k RAID samples with Focal Loss to minimize false positives).
* **Secondary Detector (Multilingual):** `mujian2026/multilingual-ai-text-detector` (XLM-RoBERTa-base derivative reserved for cross-lingual screening).
* **Third Detector:** `ShantanuT01/gradient-ai-text-detector` (DeBERTa-v3-based single-logit classifier).
* **Pipeline Flow:** `Input Validation` $\rightarrow$ `Token Chunking` $\rightarrow$ `Sequential Three-Model Inference` $\rightarrow$ `Softmax/Sigmoid Logit Normalization` $\rightarrow$ `Majority Vote` $\rightarrow$ `Standardized Schema Output`.

### Latest Evaluation

The current evaluation uses 101 samples: 51 human and 50 AI.

| Configuration | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Gradient alone | 82.18% | 100.00% | 64.00% | 0.7805 |
| Three-model majority | 84.16% | 84.00% | 84.00% | 0.8400 |

The agent loads the three models sequentially to reduce peak memory usage. A
two-out-of-three vote determines `likely_ai_generated` or `likely_human`, and
the result reports each model score, vote counts, and agreement strength.

## Repository Structure

```text
multimodal-ai-content-detection-agent/
│
├── docs/
├── notebooks/
├── src/
│   └── detectors/
│       └── text/
│           ├── __init__.py
│           ├── detector.py
│           ├── preprocessing.py
│           └── schemas.py
│
├── tests/
│   └── test_text_detector.py
│
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

## Quickstart & Testing
### Install Dependencies:
```Bash
pip install -r requirements.txt
```

### Run Standardized Tests:
```Bash
pytest -s
```

### Run the Three-Model Agent:
```Bash
python3 scripts/test_text_agent.py
```

The project uses a virtual environment in `.venv`. Activate it first when the
system Python does not contain the required dependencies:

```Bash
source .venv/bin/activate
```

Detailed evaluation outputs are written to `results/`, including
`extended_gradient_results.csv` and `three_models_comparison.csv`.

### Development Approach
Learn $\rightarrow$ Experiment $\rightarrow$ Implement $\rightarrow$ Evaluate $\rightarrow$ Integrate $\rightarrow$ Deploy