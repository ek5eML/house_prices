# House Prices

## Setup

Create a virtual environment and install dependencies from `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place Kaggle data files in the `data/` directory:

```
data/train.csv
data/test.csv
```

Run the pipeline:

```bash
python main.py
```

Configure the model and run mode in `config.py` (`training_model`, `mode`).

## Workflow

Set the run mode in `config.py` via `mode`. Choose the model via `training_model`.

### `train` — cross-validation

- Loads the training set and runs **KFold CV** (5 folds by default) on **all** training data.
- Computes the mean metric and std across folds, and writes the result to `logs/{experiment_name}.txt`.
- If `save_best_model: True`, updates `logs/{model}.txt` with the best parameters (when the CV score improves).

The model is **not** saved to checkpoints in this mode.

### `fit` — training and saving

- Loads the training set and splits it into **train/val** (`val_size: 0.2`, i.e. 80% / 20%) — **not on the full dataset**.
- Trains the pipeline on the train split, evaluates the metric on val, and logs to `logs/{experiment_name}.txt`.
- Saves the trained pipeline to `checkpoints/{training_model}.joblib`.

### `submit` — test prediction

- Loads the test set.
- Loads the saved model from `checkpoints/{training_model}.joblib` and runs `predict`.
- Writes the output to `submission.csv`.

Exception: for `xgboost`, `lightgbm`, `ensemble`, or when `rerun: True`, the model is **retrained** via `fit` before prediction (instead of loading from checkpoint).

## Results

| Approach            | CV        | CV STD   | LB      | Date       |
| ------------------- | --------- | -------- | ------- | ---------- |
| ensemble (stacking) | -0.014551 | 0.002749 | 0.12274 | 2026-06-29 |
| ensemble (voting)   | -0.014197 | 0.002501 | 0.12311 | 2026-06-28 |
| catboost            | -0.014496 | 0.002782 | 0.12408 | 2026-06-28 |
| xgboost             | -0.015429 | 0.002542 | 0.12723 | 2026-06-28 |
| lightgbm            | -0.015512 | 0.002317 | 0.12997 | 2026-06-28 |
| elasticnet          | -0.020482 | 0.007729 | 0.13391 | 2026-06-28 |
| ridge               | -0.020336 | 0.007526 | 0.13409 | 2026-06-27 |
| lasso               | -0.023697 | 0.008131 | 0.13474 | 2026-06-27 |
| random_forest       | -0.018612 | 0.002526 | 0.14010 | 2026-06-28 |
| DNN                 | -0.019884 | 0.004727 | 0.14251 | 2026-06-28 |
| decision_tree       | -0.034155 | 0.002154 | 0.19242 | 2026-06-28 |
| regression          | -0.038462 | 0.017765 | 0.21016 | 2026-06-28 |
| KNN                 | -0.045571 | 0.006306 | 0.22028 | 2026-06-27 |

