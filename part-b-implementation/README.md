# Part B - Minimal Model Training Pipeline

This is a small, reproducible training pipeline for Option 2 of the take-home.

## Task

I used the SkillsFuture unique skills list and converted the workbook's `Unique Skills List` sheet into `data/raw/skillsfuture_unique_skills.csv`. The task is binary text classification:

> Given a skill title and description, predict whether the skill is marked as an `Emerging Skills` item.

This keeps the pipeline small while still giving us a real class imbalance problem: only about 9% of rows are marked as emerging.

## Design Notes

- **Dataset versioning:** the pipeline writes a `dataset_manifest.json` containing the source file path, SHA-256 checksum, row count, and label distribution.
- **Baseline:** TF-IDF features plus logistic regression. I use this as the cheap model the candidate has to beat.
- **Candidate model:** frozen MiniLM sentence embeddings with a trainable linear classification head. This is closer to a lightweight head-tuning setup than a bag-of-words model, while still running comfortably on CPU.
- **Reproducibility:** the train/validation/test split and model seeds are controlled by `--seed`.
- **Metrics:** `metrics.json` includes positive-class precision/recall/F1, macro F1, weighted F1, average precision, prevalence, split sizes, thresholds, seed, and model settings.
- **Threshold tuning:** both models choose a decision threshold on the validation split, then report final metrics on the held-out test split.
- **Promotion:** the candidate is promoted only if it beats the TF-IDF baseline on positive-class F1 and recall, with positive recall at least `0.40`. Weighted F1 is logged, but it is not the promotion metric because the majority class is much easier.
- **With more time:** I would add cross-validation, threshold calibration, error analysis by sector/skill family, and optionally unfreeze the final encoder layer if compute and governance constraints allowed it.

## Methodology

I treat this as an imbalanced binary classification problem. The input text is:

```text
<skill title>. <skill description>
```

The label is `Emerging Skills`, converted to `0/1`. The dataset is split into train/validation/test with stratification and a fixed seed. The validation split is used only to choose the decision threshold; final reported metrics are from the held-out test split.

The candidate architecture is:

```text
skill title + description
  -> frozen MiniLM sentence embedding model
  -> trainable logistic classification head
  -> probability of Emerging Skills
```

This is a small head-tuning pipeline rather than full transformer fine-tuning. I chose it because the dataset is small, the task is straightforward, and the prompt says GPU access is not the evaluation focus. Freezing the encoder also makes the run faster and more reproducible while still using a learned language representation.

The TF-IDF/logistic regression baseline is included to keep promotion honest: a candidate model should not be considered ready just because it trains successfully. It has to beat the baseline on the minority class, which is the class that matters for this task.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Run Training

```powershell
python -m skill_emerging_pipeline.train --seed 42
```

Outputs are written to `artifacts/`:

- `dataset_manifest.json`
- `metrics.json`
- `promotion.json`
- `baseline_classification_report.txt`
- `candidate_classification_report.txt`
- `classification_head.joblib`

With `--seed 42`, the current run produces:

Baseline TF-IDF/logistic regression:

- average precision: `0.371`
- positive precision: `0.357`
- positive recall: `0.333`
- positive F1: `0.345`
- macro F1: `0.641`
- weighted F1: `0.883`

Candidate frozen-embedding/classification-head model:

- average precision: `0.439`
- positive precision: `0.522`
- positive recall: `0.400`
- positive F1: `0.453`
- macro F1: `0.703`
- weighted F1: `0.907`
- promotion: `true`

## Run Tests

```powershell
pytest
```

The tests cover:

- text feature construction from skill title and description, including missing descriptions
- metric calculation for positive-class, macro, and weighted scores
- promotion logic against a baseline model
- reproducible training of the frozen-embedding classification head with the same seed
- validation-threshold selection while respecting a minimum recall constraint
