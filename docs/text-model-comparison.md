# Text AI Detection Model Comparison

## Overview
This document evaluates candidate models for the Text AI Detection module of the Multimodal AI Content Detection System. The evaluation balances accuracy, false-positive rates, language coverage, and hardware constraints suitable for GitHub Codespaces, Google Colab, and eventual web deployment.

---

## Project Constraints & Technical Requirements

- **Model Source:** Pretrained models only (no fine-tuning in V1 baseline).
- **Tooling:** Free and open-source deployment tools (Transformers, ONNX Runtime, PyTorch).
- **Environment:** Low-resource local execution / GitHub Codespaces + Google Colab for initial testing.
- **Reliability Target:** Low false-positive rate on human-authored text.
- **Standardized Output:** Normalization to probability scores (`ai_score` vs. `human_score`).

---

## Candidate 1: Multilingual AI Text Detector

* **Repository:** `mujian2026/multilingual-ai-text-detector`
* **Architecture:** XLM-RoBERTa-base (~279M parameters)
* **Task:** Binary AI vs. Human text classification
* **Supported Languages:** English, Simplified Chinese, Vietnamese (documented); experimental for others.

### Evaluation & Benchmark Summary
* **Training Data:** 900 QA pairs (300 English, 300 Chinese, 300 Vietnamese) generated using `Qwen2.5-1.5B-Instruct`.
* **Reported Performance:**
  * English F1: `0.9890`
  * Vietnamese F1: `0.9783`
  * Chinese F1: `0.9462`
  * Overall F1: `0.9710`
* **Available Quantization:** ONNX variants including `q4` (~181 MB), `q8` (~279 MB), and `fp32` (~1.11 GB).

### Strengths
- Native multilingual capability with a unified XLM-R architecture.
- Optimized ONNX variants ready for browser-side or lightweight CPU runtime.
- Low memory footprint in `q4` format.

### Weaknesses & Limitations
- **Small Training Dataset:** 900 samples limit domain coverage and generalization across distinct prompt styles.
- **Generator Bias:** AI training set relies heavily on a single generator family (`Qwen2.5`). Performance on GPT-4, Claude, or Gemini outputs is unverified.
- **Sequence Length:** Upstream setup limited max sequence length to 256 tokens.

---

## Candidate 2: TMR AI Text Detector (Primary Choice)

* **Repository (ONNX):** `onnx-community/tmr-ai-text-detector-ONNX`
* **Original Model:** `Oxidane/tmr-ai-text-detector`
* **Architecture:** RoBERTa-base (~125M parameters)
* **Task:** Binary AI vs. Human text classification
* **Primary Language:** English

### Evaluation & Benchmark Summary
* **Training Data:** 50,000 stratified samples from the RAID benchmark (45% Human / 55% AI). Uses **Focal Loss** and **Self-Hard-Negative Mining** to minimize false positives.
* **Reported Performance (RAID Benchmark):**
  * AUROC: `99.28%`
  * TPR @ 5% FPR: `95.79%`
  * TPR @ 1% FPR: `90.17%`
* **Held-out Split Metrics:** AUROC `99.69%`, Accuracy `97.42%`, FPR `2.61%`, FNR `2.58%`.
* **Available Quantization:** `model_quantized.onnx` (~126 MB), `model_int8.onnx` (~126 MB), `model_q4.onnx` (~212 MB), `fp32` (~499 MB).

### Strengths
- Robust training data scale (50k RAID samples across varied generators and domains).
- Explicit loss tuning focused on reducing false positives on human text.
- Standard 512-token context window support.
- Smaller base architecture (125M parameters vs. XLM-R's 279M).

### Weaknesses & Limitations
- **Language Limitation:** English-focused; performance degrades on out-of-distribution non-English text.
- **Short Text Vulnerability:** Accuracy drops on brief or conversational snippets (<10 words).
- **Adversarial Noise:** Vulnerable to heavy paraphrasing or intentional typo insertion.

---

## Direct Model Comparison Matrix

| Requirement / Metric | Multilingual Detector (`mujian2026`) | TMR Detector (`Oxidane` / `onnx-community`) |
| :--- | :--- | :--- |
| **Primary Focus** | Multilingual Screening | English AI Text Detection |
| **Base Model** | XLM-RoBERTa-base | RoBERTa-base |
| **Parameter Count** | ~279 million | ~125 million |
| **Training Dataset** | 900 QA pairs (Qwen2.5) | 50,000 RAID samples |
| **False-Positive Mitigation** | Baseline cross-entropy | Focal Loss + Hard Negative Mining |
| **Max Sequence Length** | 256 tokens | 512 tokens |
| **Quantized Model Size** | ~181 MB (`q4`) | ~126 MB (`int8` / `quantized`) |
| **Primary Use Case** | Cross-lingual screening signal | High-precision English detector |

---

## Final Architecture & Model Selection

### Strategic Decision: Dual-Tiered Deployment

1. **Primary English Detector:** `Oxidane/tmr-ai-text-detector`
   * **Role:** Default inference engine for English input. Selected due to superior training dataset size, lower parameter count, 512-token context length, and benchmark performance designed to reduce false positives.

2. **Secondary Multilingual Detector:** `mujian2026/multilingual-ai-text-detector`
   * **Role:** Fallback engine for multilingual screening (Chinese, Vietnamese, experimental cross-lingual evaluation).

### Execution Strategy for V1 Baseline
- Begin implementation using PyTorch and `transformers` on the primary model (`Oxidane/tmr-ai-text-detector`) to validate tokenization, text cleaning, logit processing, and schema formatting.
- Transition inference to lightweight ONNX Runtime once PyTorch baseline validation and unit tests pass.