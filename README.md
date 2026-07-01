# Internal Docs QA Pipeline

This repo contains a compact design and implementation exercise around internal document QA and lightweight model promotion.

## Contents

- `part-a-system-design.md` - on-prem internal document Q&A system design
- `part-b-implementation/` - reproducible emerging-skills classification pipeline
- `part-c-reflection.md` - short production investigation note

## Part B Quick Start

```powershell
cd .\part-b-implementation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
python -m skill_emerging_pipeline.train --seed 42
pytest
```

The model pipeline writes dataset, metric, and promotion artifacts under `part-b-implementation/artifacts/`.
