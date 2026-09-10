# Text Detector Error Analysis

## Models Evaluated

- `mujian2026/multilingual-ai-text-detector`
- `Oxidane/tmr-ai-text-detector`
- `ShantanuT01/gradient-ai-text-detector`

*No fine-tuning performed. All three models were evaluated using pretrained weights.*

---

## Dataset

**101 samples total:**
- Human: 51
- AI: 50

Same dataset used for all three models and the ensemble comparison.

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

## Best Individual Model Performance Comparison

| Model | Best Accuracy | Best F1 | Precision | Recall | FP | FN |
|-------|--------------:|--------:|----------:|-------:|---:|---:|
| **TMR** | 73.27% @ 0.95 | **76.80%** @ 0.70 | 73.47% | 72.00% | 13 | 14 |
| **Multilingual** | **75.25%** @ 0.50/0.60 | 74.23% @ 0.50/0.60 | **76.60%** | 72.00% | **11** | 14 |
| **Gradient** | **82.18%** @ 0.50 | 78.05% @ 0.50 | **100.00%** | 64.00% | **0** | 18 |

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

### Gradient — `ShantanuT01/gradient-ai-text-detector`

Gradient was evaluated at threshold `0.50`. Its single output logit is
converted with sigmoid before classification.

| Accuracy | Precision | Recall | F1 Score | FP | FN |
|---------:|----------:|-------:|---------:|---:|---:|
| 82.18% | 100.00% | 64.00% | 78.05% | 0 | 18 |

- True positives: 32
- True negatives: 51
- The model avoided false positives but missed 18 AI samples on this dataset.

### Three-Model Majority Ensemble

The agent runs TMR, multilingual, and Gradient sequentially, then assigns the
class with at least two votes.

| Metric | Result |
|---|---:|
| Full three-way agreement | 49 / 101 |
| Accuracy | 84.16% |
| Precision | 84.00% |
| Recall | 84.00% |
| F1 Score | 0.8400 |
| True Positives | 42 |
| False Positives | 8 |
| True Negatives | 43 |
| False Negatives | 8 |
| Uncertain predictions | 0 |

Among the 34 TMR/multilingual disagreements, Gradient voted AI in 6 cases and
human in 28 cases.

---

## Comparison with Earlier 22-Sample Evaluation

| Model | 22-Sample Accuracy | 101-Sample Accuracy |
|-------|-------------------:|--------------------:|
| TMR | 68.18% | 73.27% |
| Multilingual | 95.45% | 75.25% |

The multilingual model's performance dropped significantly, demonstrating that smaller evaluations can be overly optimistic.

---

## Key Findings

1. **Three-model majority:** Highest measured accuracy (**84.16%**) and F1
	(**0.8400**) on the current evaluation set.
2. **Gradient:** Highest measured precision (**100.00%**) and no false
	positives, but lower recall (**64.00%**).
3. **TMR:** Highest individual F1 among the earlier two models (**76.80%** at
	threshold `0.70`).
4. **Multilingual:** Strongest individual accuracy among the earlier two models
	(**75.25%**).
5. **No detector** provides definitive evidence of AI authorship.
6. **22-sample evaluation** was misleading; the 101-sample results are more
	informative but still limited.

---

## Model Selection Recommendation

| Priority | Recommended Model | Threshold |
|----------|-------------------|-----------|
| Best current aggregate result | Three-model majority | TMR 0.70, Multilingual 0.50, Gradient 0.50 |
| Highest individual precision | Gradient | 0.50 |
| Higher F1 among the original two models | TMR | 0.70 |

**All three models remain candidates within the ensemble.** Further validation on larger, more diverse datasets is required.

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
python3 scripts/dev/inspect_text_detector_2.py
python3 scripts/dev/inspect_text_detector_3.py
python3 scripts/evaluate_text_detector_3.py
python3 scripts/compare_text_models.py
python3 scripts/test_text_agent.py
```

CSV outputs saved to `results/` directory. HF_TOKEN optional (higher rate limits with authentication).

---

## Conclusion

The expanded 101-sample evaluation provides a more reliable basis for comparison than the original 22-sample evaluation.

- **Best individual accuracy:** Gradient (82.18%)
- **Best individual F1 among TMR and multilingual:** TMR (76.80% @ threshold 0.70)
- **Best current aggregate result:** Three-model majority (84.16% accuracy,
  0.8400 F1)

* No detector is definitive.
* Next step: **validation on a larger, more diverse, independently held-out
	dataset** before making stronger model-selection claims.