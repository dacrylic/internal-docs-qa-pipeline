import pandas as pd
import numpy as np

from skill_emerging_pipeline.data import make_model_text
from skill_emerging_pipeline.train import (
    compute_metrics,
    find_best_threshold,
    promotion_decision,
    train_embedding_head,
)


def test_make_model_text_combines_title_and_description():
    text = make_model_text("Data Analysis", "Interpret datasets for business decisions")

    assert text == "Data Analysis. Interpret datasets for business decisions"


def test_make_model_text_handles_missing_description():
    text = make_model_text("Data Analysis", pd.NA)

    assert text == "Data Analysis."


def test_promotion_decision_requires_recall_and_baseline_lift():
    candidate = {"positive_f1": 0.50, "average_precision": 0.45, "positive_recall": 0.30}
    baseline = {"positive_f1": 0.40, "average_precision": 0.40, "positive_recall": 0.20}

    decision = promotion_decision(candidate, baseline)

    assert decision["promoted"] is False
    assert decision["checks"]["candidate_positive_f1_beats_baseline"] is True
    assert decision["checks"]["candidate_positive_recall_beats_baseline"] is True
    assert decision["checks"]["candidate_positive_recall_at_least_0_40"] is False


def test_compute_metrics_returns_classification_metrics():
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = compute_metrics(labels, predictions, probabilities)

    assert metrics["positive_precision"] == 2 / 3
    assert metrics["positive_recall"] == 1.0
    assert metrics["positive_f1"] == 0.8
    assert metrics["macro_f1"] > 0
    assert metrics["weighted_f1"] > 0
    assert metrics["positive_prevalence"] == 0.5


def test_train_head_is_reproducible_with_same_seed():
    features = np.array(
        [
            [1.0, 0.0, 0.2],
            [0.9, 0.1, 0.1],
            [0.0, 1.0, 0.8],
            [0.1, 0.9, 0.9],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.float32)

    model_a = train_embedding_head(features, labels, seed=7)
    model_b = train_embedding_head(features, labels, seed=7)

    probs_a = model_a.predict_proba(features)[:, 1]
    probs_b = model_b.predict_proba(features)[:, 1]

    assert np.allclose(probs_a, probs_b)


def test_find_best_threshold_respects_minimum_recall():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, metrics = find_best_threshold(labels, probabilities, min_recall=1.0)

    assert threshold <= 0.6
    assert metrics["positive_recall"] == 1.0
