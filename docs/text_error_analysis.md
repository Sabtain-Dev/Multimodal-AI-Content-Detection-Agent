# Text Detector Error Analysis

## Models Evaluated

- `mujian2026/multilingual-ai-text-detector`
- `Oxidane/tmr-ai-text-detector`

*No fine-tuning performed. Both models evaluated using pretrained weights.*

---

## Dataset

**101 samples total:**
- Human: 51
- AI: 50

Same dataset used for both models.

---

## Classification Method

- AI score `>= threshold` → **AI**
- AI score `< threshold` → **HUMAN**
- Thresholds evaluated: 0.50 to 0.95

---

## Results

### TMR — `Oxidane/tmr-ai-text-detector`

| Threshold | Accuracy | Precision | Recall | F1 Score | FP | FN |
|----------:|---------:|----------:|-------:|---------:|---:|---:|
| 0.50 | 68.32% | 60.98% | 100.00% | 75.76% | 32 | 0 |
| 0.60 | 69.31% | 62.03% | 98.00% | 75.97% | 30 | 1 |
| **0.70** | **71.29%** | **64.00%** | **96.00%** | **76.80%** | 27 | 2 |
| 0.80 | 71.29% | 65.67% | 88.00% | 75.21% | 23 | 6 |
| 0.90 | 70.30% | 66.13% | 82.00% | 73.21% | 21 | 9 |
| **0.95** | **73.27%** | **73.47%** | 72.00% | 72.73% | 13 | 14 |

- **Best Accuracy:** 73.27% @ threshold 0.95
- **Best F1:** 76.80% @ threshold 0.70

---

### Multilingual — `mujian2026/multilingual-ai-text-detector`

| Threshold | Accuracy | Precision | Recall | F1 Score | FP | FN |
|----------:|---------:|----------:|-------:|---------:|---:|---:|
| **0.50** | **75.25%** | **76.60%** | **72.00%** | **74.23%** | 11 | 14 |
| **0.60** | **75.25%** | **76.60%** | **72.00%** | **74.23%** | 11 | 14 |
| 0.70 | 74.26% | 76.09% | 70.00% | 72.92% | 11 | 15 |
| 0.80 | 74.26% | 76.09% | 70.00% | 72.92% | 11 | 15 |
| 0.90 | 74.26% | 76.09% | 70.00% | 72.92% | 11 | 15 |
| 0.95 | 73.27% | 75.56% | 68.00% | 71.58% | 11 | 16 |

- **Best Accuracy:** 75.25% @ thresholds 0.50–0.60
- **Best F1:** 74.23% @ thresholds 0.50–0.60

---

## Best Performance Comparison

| Model | Best Accuracy | Best F1 | Precision | Recall | FP | FN |
|-------|--------------:|--------:|----------:|-------:|---:|---:|
| **TMR** | 73.27% @ 0.95 | **76.80%** @ 0.70 | 73.47% | 72.00% | 13 | 14 |
| **Multilingual** | **75.25%** @ 0.50/0.60 | 74.23% @ 0.50/0.60 | **76.60%** | 72.00% | **11** | 14 |

---

## Error Summary

### TMR
- **False Positives:** 13–32 (decreases as threshold increases)
- **False Negatives:** 0–14 (increases as threshold increases)
- High AI scores on human samples observed (some human samples scored ~0.98)

### Multilingual
- **False Positives:** 11 (stable across thresholds)
- **False Negatives:** 14–16 (slight increase at higher thresholds)
- Fewer false positives than TMR
- Misses more AI samples than TMR at lower thresholds

---

## Comparison with Earlier 22-Sample Evaluation

| Model | 22-Sample Accuracy | 101-Sample Accuracy |
|-------|-------------------:|--------------------:|
| TMR | 68.18% | 73.27% |
| Multilingual | 95.45% | 75.25% |

The multilingual model's performance dropped significantly, demonstrating that smaller evaluations can be overly optimistic.

---

## Key Findings

1. **Multilingual detector:** Highest accuracy (**75.25%**)
2. **TMR:** Highest F1 score (**76.80%** @ threshold 0.70)
3. **Multilingual:** Fewer false positives (11 vs 13 minimum for TMR)
4. **TMR:** Higher recall at lower thresholds but more false positives
5. **Neither model** provides definitive evidence of AI authorship
6. **22-sample evaluation** was misleading; 101-sample results are more reliable

---

## Model Selection Recommendation

| Priority | Recommended Model | Threshold |
|----------|-------------------|-----------|
| Higher accuracy & fewer false positives | Multilingual | 0.50–0.60 |
| Higher F1 & stronger AI recall | TMR | 0.70 |

**Both models should remain candidates.** Further validation on larger, more diverse datasets is required.

---

## Limitations

- Small dataset (101 samples)
- May not represent all domains, AI systems, or languages
- Thresholds optimized on same evaluation data (not independently validated)
- Results describe performance on this dataset, not universal accuracy

---

## Reproduction

```bash
source .venv/bin/activate
python3 scripts/inspect_text_detector_2.py
python3 scripts/compare_text_models.py
```

CSV outputs saved to `results/` directory. HF_TOKEN optional (higher rate limits with authentication).

---

## Conclusion

The expanded 101-sample evaluation provides a more reliable basis for comparison than the original 22-sample evaluation.

- **Best Accuracy:** Multilingual (75.25%)
- **Best F1:** TMR (76.80% @ threshold 0.70)

* Neither model is definitive. 
* Next step: **detailed error analysis and validation on larger, more diverse dataset** before final model selection.