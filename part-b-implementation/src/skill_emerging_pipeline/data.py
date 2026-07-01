from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "parent_skill_title",
    "parent_skill_description",
    "Emerging Skills",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_model_text(title: object, description: object) -> str:
    title_text = "" if pd.isna(title) else str(title).strip()
    description_text = "" if pd.isna(description) else str(description).strip()
    return f"{title_text}. {description_text}".strip()


def load_skills_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name="Unique Skills List")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dataset = pd.DataFrame(
        {
            "text": [
                make_model_text(title, description)
                for title, description in zip(
                    df["parent_skill_title"],
                    df["parent_skill_description"],
                )
            ],
            "label": df["Emerging Skills"].astype(bool).astype(int),
        }
    )
    dataset = dataset[dataset["text"].str.len() > 0].reset_index(drop=True)
    if dataset["label"].nunique() < 2:
        raise ValueError("Dataset must contain both positive and negative labels.")
    return dataset


def write_dataset_manifest(source_path: Path, dataset: pd.DataFrame, output_path: Path) -> None:
    manifest = {
        "source_file": str(source_path),
        "source_sha256": file_sha256(source_path),
        "rows": int(len(dataset)),
        "positive_rows": int(dataset["label"].sum()),
        "negative_rows": int((dataset["label"] == 0).sum()),
        "label": "Emerging Skills",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
