from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skill_emerging_pipeline.data import load_skills_dataset, write_dataset_manifest


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    encoder = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_baseline_model(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=8000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def train_embedding_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    seed: int,
) -> LogisticRegression:
    set_seed(seed)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(train_features, train_labels)
    return model


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "positive_precision": float(precision_score(labels, predictions, zero_division=0)),
        "positive_recall": float(recall_score(labels, predictions, zero_division=0)),
        "positive_f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "positive_prevalence": float(labels.mean()),
    }


def find_best_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.40,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.50
    best_metrics = compute_metrics(labels, (probabilities >= best_threshold).astype(int), probabilities)
    best_key = (-1.0, -1.0)

    for threshold in np.linspace(0.05, 0.95, 91):
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_metrics(labels, predictions, probabilities)
        if metrics["positive_recall"] < min_recall:
            continue
        key = (metrics["positive_f1"], metrics["average_precision"])
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def promotion_decision(
    candidate_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, object]:
    checks = {
        "candidate_positive_f1_beats_baseline": candidate_metrics["positive_f1"]
        >= baseline_metrics["positive_f1"],
        "candidate_positive_recall_beats_baseline": candidate_metrics["positive_recall"]
        >= baseline_metrics["positive_recall"],
        "candidate_positive_recall_at_least_0_40": candidate_metrics["positive_recall"] >= 0.40,
    }
    return {
        "promoted": all(checks.values()),
        "checks": checks,
        "rationale": "Promote when the candidate improves positive-class F1 and recall over the baseline and clears the recall floor.",
    }


def evaluate_baseline(
    train_texts: list[str],
    train_labels: np.ndarray,
    validation_texts: list[str],
    validation_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    seed: int,
) -> tuple[Pipeline, dict[str, float], dict[str, float], np.ndarray, float]:
    baseline = build_baseline_model(seed)
    baseline.fit(train_texts, train_labels)
    validation_probabilities = baseline.predict_proba(validation_texts)[:, 1]
    threshold, validation_metrics = find_best_threshold(validation_labels, validation_probabilities)
    test_probabilities = baseline.predict_proba(test_texts)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_metrics = compute_metrics(test_labels, test_predictions, test_probabilities)
    return baseline, validation_metrics, test_metrics, test_predictions, threshold


def evaluate_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    seed: int,
) -> tuple[LogisticRegression, dict[str, float], dict[str, float], np.ndarray, float]:
    model = train_embedding_head(train_features, train_labels, seed)
    validation_probabilities = model.predict_proba(validation_features)[:, 1]
    threshold, validation_metrics = find_best_threshold(validation_labels, validation_probabilities)
    test_probabilities = model.predict_proba(test_features)[:, 1]
    test_predictions = (test_probabilities >= threshold).astype(int)
    test_metrics = compute_metrics(test_labels, test_predictions, test_probabilities)
    return model, validation_metrics, test_metrics, test_predictions, threshold


def run_pipeline(
    data_path: Path,
    output_dir: Path,
    seed: int,
    embedding_model: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_skills_dataset(data_path)
    write_dataset_manifest(data_path, dataset, output_dir / "dataset_manifest.json")

    train_df, holdout_df = train_test_split(
        dataset,
        test_size=0.3,
        random_state=seed,
        stratify=dataset["label"],
    )
    validation_df, test_df = train_test_split(
        holdout_df,
        test_size=0.5,
        random_state=seed,
        stratify=holdout_df["label"],
    )

    train_features = embed_texts(train_df["text"].tolist(), embedding_model)
    validation_features = embed_texts(validation_df["text"].tolist(), embedding_model)
    test_features = embed_texts(test_df["text"].tolist(), embedding_model)
    train_labels = train_df["label"].to_numpy(dtype=np.float32)
    validation_labels = validation_df["label"].to_numpy(dtype=np.int64)
    test_labels = test_df["label"].to_numpy(dtype=np.int64)
    train_texts = train_df["text"].tolist()
    validation_texts = validation_df["text"].tolist()
    test_texts = test_df["text"].tolist()

    (
        _baseline_model,
        baseline_validation_metrics,
        baseline_test_metrics,
        baseline_predictions,
        baseline_threshold,
    ) = evaluate_baseline(
        train_texts=train_texts,
        train_labels=train_labels.astype(np.int64),
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        seed=seed,
    )

    (
        candidate_model,
        candidate_validation_metrics,
        candidate_test_metrics,
        predictions,
        candidate_threshold,
    ) = evaluate_candidate(
        train_features=train_features,
        train_labels=train_labels.astype(np.int64),
        validation_features=validation_features,
        validation_labels=validation_labels,
        test_features=test_features,
        test_labels=test_labels,
        seed=seed,
    )
    candidate_test_metrics.update(
        {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "seed": seed,
            "embedding_model": embedding_model,
            "model": "frozen_embedding_logistic_head",
            "threshold": candidate_threshold,
        }
    )
    baseline_test_metrics.update(
        {
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
            "seed": seed,
            "model": "tfidf_logistic_regression",
            "threshold": baseline_threshold,
        }
    )

    run_metrics = {
        "baseline_validation": baseline_validation_metrics,
        "baseline": baseline_test_metrics,
        "candidate_validation": candidate_validation_metrics,
        "candidate": candidate_test_metrics,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "seed": seed,
    }
    decision = promotion_decision(candidate_test_metrics, baseline_test_metrics)

    (output_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2), encoding="utf-8")
    (output_dir / "promotion.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (output_dir / "candidate_classification_report.txt").write_text(
        classification_report(test_labels, predictions, zero_division=0),
        encoding="utf-8",
    )
    (output_dir / "baseline_classification_report.txt").write_text(
        classification_report(test_labels, baseline_predictions, zero_division=0),
        encoding="utf-8",
    )
    joblib.dump(
        {
            "classification_head": candidate_model,
            "embedding_model": embedding_model,
            "seed": seed,
            "threshold": candidate_threshold,
        },
        output_dir / "classification_head.joblib",
    )
    return {"metrics": run_metrics, "promotion": decision}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an emerging-skills text classifier.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/skillsfuture_unique_skills.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_pipeline(
        data_path=args.data,
        output_dir=args.output_dir,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
