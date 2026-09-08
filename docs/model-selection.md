# Model Selection

The text detector currently uses three pretrained models. TMR provides the
English-focused signal, the multilingual model adds cross-lingual coverage,
and Gradient provides a third independent signal for majority voting.

## Selection Criteria

For every candidate model, we will evaluate:

1. Task
2. Modality
3. Architecture
4. Training dataset
5. Languages
6. Model size
7. Input requirements
8. Output format
9. Evaluation metrics
10. License
11. Inference requirements
12. Known limitations

## Candidate Model Table

| Model | Modality | Task | Size | License | Dataset | Notes |
|---|---|---|---|---|---|---|
| `Oxidane/tmr-ai-text-detector` | Text | AI Detection | RoBERTa-base, ~125M parameters | Check model card | RAID, 50k samples | English-focused; threshold `0.70` |
| `mujian2026/multilingual-ai-text-detector` | Text | AI Detection | XLM-RoBERTa-base, ~279M parameters | Check model card | 900 QA pairs | ONNX runtime; threshold `0.50` |
| `ShantanuT01/gradient-ai-text-detector` | Text | AI Detection | DeBERTa-v3-based classifier | Check model card | Check model card | Single-logit output; threshold `0.50` |
| TBD | Image | AI Detection | TBD | TBD | TBD | TBD |
| TBD | Audio | AI Detection | TBD | TBD | TBD | TBD |
| TBD | Video | AI Detection | TBD | TBD | TBD | TBD |

## Current Text Selection

The three text models are loaded sequentially to reduce memory pressure. Each
model produces a normalized `ai_score`, `human_score`, and prediction through
the shared `TextAIDetector` schema. The ensemble selects the majority class:

- Two or three AI votes produce `likely_ai_generated`.
- Two or three human votes produce `likely_human`.
- `agreement_strength` is `Strong` for unanimous votes and `Moderate` for a
  two-to-one vote.

The Gradient checkpoint returns one logit rather than two class logits. The
detector converts that value with sigmoid before applying the `0.50` threshold.

On the current 101-sample evaluation set, the three-model majority achieved
`84.16%` accuracy and `0.8400` F1, compared with `82.18%` accuracy and
`0.7805` F1 for Gradient alone. These results are dataset-specific and are not
claims of universal detector accuracy.