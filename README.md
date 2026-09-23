# Multimodal AI Content Detection Agent

**Status:** Day 8 : Text + Image Detection Pipeline Active

## Description
A multimodal system designed to detect potentially AI-generated text, images, audio, and video using open-source pretrained models.

## Project Objective
Develop a resource-efficient AI detection system leveraging pretrained models without fine-tuning in Version 1.

## Modalities Progress
* 🟢 **Text:** Three-model detector and majority-vote agent implemented
* 🟢 **Image:** Dual-model ensemble detector implemented and validated
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

### Image Detection Architecture (V1)
* **Primary Image Model:** `Ateeqq/ai-vs-human-image-detector` with threshold `0.70`.
* **Secondary Image Model:** `haywoodsloan/ai-image-detector-deploy` with threshold `0.50`.
* **Ensemble Policy:** The second model acts as the primary decision-maker; if both models disagree, the final label follows the model-2 output while confidence stays capped as `Moderate`.
* **Pipeline Flow:** `Image Input` $\rightarrow$ `Model 1 Score + Model 2 Score` $\rightarrow$ `Binary Label Derivation` $\rightarrow$ `Consensus / Fallback Decision` $\rightarrow$ `Normalized Output Payload`.

### Latest Evaluation

The project retains the historical benchmark as the official validated result, while the Kaggle-backed dataset path is now enabled for reproducible evaluation when credentials are available.

| Dataset source | Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Historical local benchmark (validated) | Gradient alone | 82.18% | 100.00% | 64.00% | 0.7805 |
| Historical local benchmark (validated) | Three-model majority | 84.16% | 84.00% | 84.00% | 0.8400 |
| Kaggle dataset path (pending credential-backed validation) | TMR | Pending | Pending | Pending | Pending |
| Kaggle dataset path (pending credential-backed validation) | Multilingual | Pending | Pending | Pending | Pending |
| Kaggle dataset path (pending credential-backed validation) | Gradient | Pending | Pending | Pending | Pending |
| Kaggle dataset path (pending credential-backed validation) | Three-model majority | Pending | Pending | Pending | Pending |

The agent loads the three text models sequentially to reduce peak memory usage. A two-out-of-three vote determines `likely_ai_generated` or `likely_human`, and the result reports each model score, vote counts, and agreement strength.

The Kaggle dataset links are now treated as the canonical data source for the evaluation scripts:
- AI vs. Human Text Detection Dataset: https://www.kaggle.com/datasets/itssabtain/ai-vs-human-text-detection-dataset
- Synthetic vs. Real Image Classifier: https://www.kaggle.com/datasets/itssabtain/image-detector-ai-vs-human

To generate the official Kaggle benchmark locally, export Kaggle credentials first or place a valid `~/.kaggle/kaggle.json` file in the environment before running the evaluation scripts. Until that is done, the previously validated numbers remain the official project benchmark.

## Repository Structure

```text
multimodal-ai-content-detection-agent/
│
├── docs/
├── scripts/
├── src/
│   └── detectors/
│       ├── image/
│       │   ├── __init__.py
│       │   ├── ensemble.py
│       │   ├── image_detectors.py
│       │   └── label_normalization.py
│       └── text/
│           ├── __init__.py
│           ├── detector.py
│           ├── preprocessing.py
│           └── schemas.py
│
├── tests/
│   ├── test_image_detector_package.py
│   └── test_text_detector.py
│
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── results/
```

## License
This project is licensed under the [Apache License 2.0](LICENSE).

## Quickstart & Testing
### Install Dependencies:
```Bash
pip install -r requirements.txt
```

### Run Standardized Tests:
```Bash
pytest -s
```

### Run the Text Detector:
```Bash
python3 scripts/text_detector/test_text_agent.py
```

### Run the Image Detector Smoke Test:
```Bash
python3 -c "from src.detectors.image import ImageDetector; print(ImageDetector().predict('test.jpg', 0.8, 0.2))"
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