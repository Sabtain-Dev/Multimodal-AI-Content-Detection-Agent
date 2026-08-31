# Multimodal AI Content Detection Agent

**Status:** Day 5 : Text AI Detector Module Baseline

## Description
A multimodal system designed to detect potentially AI-generated text, images, audio, and video using open-source pretrained models.

## Project Objective
Develop a resource-efficient AI detection system leveraging pretrained models without fine-tuning in Version 1.

## Modalities Progress
* 🟢 **Text:** Baseline PyTorch/Transformers detector implemented (`Oxidane/tmr-ai-text-detector`)
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
* **Pipeline Flow:** `Input Validation` $\rightarrow$ `Truncation (512 tokens)` $\rightarrow$ `PyTorch Inference` $\rightarrow$ `Softmax Logits` $\rightarrow$ `Standardized Schema Output`.

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
