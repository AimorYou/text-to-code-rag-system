# Data Directory

This directory is managed by DVC (Data Version Control).

## Dataset

The project uses the CodeSearchNet dataset from HuggingFace.
Data is automatically downloaded when running training.

## Structure

```
data/
├── raw/          # Raw downloaded data (managed by DVC)
└── processed/    # Preprocessed data (managed by DVC)
```

## Usage

Data will be automatically downloaded on first training run.
If you need to manually pull data:

```bash
dvc pull
```
