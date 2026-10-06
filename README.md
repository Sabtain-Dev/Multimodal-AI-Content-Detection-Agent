# Multimodal AI Content Detection Agent

**Status:** Text, image, and audio detector logic plus file-based multimodal routing are implemented; video detection is planned.

## Description
A multimodal system designed to detect potentially AI-generated text, images, and audio using open-source pretrained models, with video detection planned.

## Project Objective
Develop a resource-efficient AI detection system leveraging pretrained models without fine-tuning in Version 1.

## Modalities Progress
* 🟢 **Text:** Three-model detector and majority-vote agent implemented
* 🟢 **Image:** Dual-model ensemble decision logic implemented; the multimodal agent currently accepts model scores rather than running image-model inference itself
* 🟢 **Audio:** Two-model detector, primary-decision ensemble, CLI pipeline, tests, and evaluation tooling implemented
* ⚪ **Video:** Planned

## Current Multimodal Agent

The file-based agent routes `.txt`, `.md`, and `.csv` files to text detection;
`.jpg`, `.jpeg`, `.png`, and `.webp` files to image decision logic; and `.wav`,
`.mp3`, `.ogg`, `.m4a`, and `.flac` files to audio detection. Video files are
not supported yet. Text and audio routes call their detector pipelines. The
image route combines two supplied model scores; it does not currently load an
image or run either image model itself.

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

### Audio Detection Architecture (V1)
* **Supporting Model:** `Sayantan090/audio-fake-detector` (`FAKE` → `AI`; `REAL` → `HUMAN`).
* **Primary Model:** `Mahmoud59/wav2vec2-fake-audio-detector` (`LABEL_0` → `AI`; `LABEL_1` → `HUMAN`), using a `0.90` AI-score threshold.
* **Ensemble Policy:** Model 1 contributes telemetry and agreement information; Model 2 determines the final label using the threshold.
* **Pipeline Flow:** `Audio File(s)` $\rightarrow$ `Model 1 Inference` $\rightarrow$ `Model 2 Inference` $\rightarrow$ `Model 2 Threshold Decision` $\rightarrow$ `Confidence + Agreement Telemetry` $\rightarrow$ `Structured Result`.

### Latest Evaluation

The project retains the previously reported benchmark results below. The linked
Kaggle text and image datasets are the evaluation datasets already used for
project testing; another direct Kaggle download or evaluation run is not
required.

| Dataset source | Configuration | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Historical local benchmark (validated) | Gradient alone | 82.18% | 100.00% | 64.00% | 0.7805 |
| Historical local benchmark (validated) | Three-model majority | 84.16% | 84.00% | 84.00% | 0.8400 |

The agent loads the three text models sequentially to reduce peak memory usage. A two-out-of-three vote determines `likely_ai_generated` or `likely_human`, and the result reports each model score, vote counts, and agreement strength.

### Evaluation Datasets

These are the Kaggle dataset collections used for text and image evaluation:

- [AI vs. Human Text Detection Dataset](https://www.kaggle.com/datasets/itssabtain/ai-vs-human-text-detection-dataset)
- [Synthetic vs. Real Image Classifier](https://www.kaggle.com/datasets/itssabtain/image-detector-ai-vs-human)

The evaluation scripts first honor `MULTIMODAL_TEXT_DATASET_DIR` or
`MULTIMODAL_IMAGE_DATASET_DIR`, then use `data/<modality>/evaluation` when
local evaluation data is present. If it is absent, the shared dataset resolver
can download the corresponding Kaggle dataset and place its files under that
local data path.

## Repository Structure

```text
multimodal-ai-content-detection-agent/
│
├── docs/
├── scripts/
├── src/
│   └── detectors/
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── audio_detectors.py
│       │   ├── ensemble.py
│       │   └── label_normalization.py
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
│   ├── test_audio_detector_package.py
│   ├── test_audio_label_normalization.py
│   ├── test_image_detector_package.py
│   └── test_text_detector.py
│
├── .gitignore
├── pytest.ini
├── README.md
├── requirements.txt
└── results/
```

See [docs/architecture.md](docs/architecture.md) for the current routing
overview and [docs/audio-detector.md](docs/audio-detector.md) for the audio
execution graph, output contract, test commands, and evaluation artifacts.

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

### Run the Audio Detector:
```Bash
python3 scripts/audio_detector/pipeline.py --input path/to/audio.wav
```

Run the audio unit tests with:
```Bash
python3 -m pytest tests/test_audio_label_normalization.py tests/test_audio_detector_package.py -q
```

See [docs/audio-detector.md](docs/audio-detector.md) for the complete audio output schema, execution diagram, test commands, and evaluation artifact list.

The project uses a virtual environment in `.venv`. Activate it first when the
system Python does not contain the required dependencies:

```Bash
source .venv/bin/activate
```

Detailed evaluation outputs are written to `results/`, including
`extended_gradient_results.csv` and `three_models_comparison.csv`.

### Development Approach
Learn $\rightarrow$ Experiment $\rightarrow$ Implement $\rightarrow$ Evaluate $\rightarrow$ Integrate $\rightarrow$ Deploy