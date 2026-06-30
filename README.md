# House Prices

Kaggle [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) project with a reproducible training pipeline.

## Setup

Recommended: create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
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

On startup, `main.py` quietly installs missing dependencies from `requirements.txt`.

Configure the run in `config.py`:

- `mode` — `train` or `fit`
- `training_model` — model name for `train` mode
- `models_to_evaluate` — list of models for `fit` mode
- `model_type` — `regression` or `classification`
- `need_scaler` — apply `StandardScaler` to features (sklearn and DNN)
- `models_params` — hyperparameters per model

EDA and experiments: `research.ipynb`.

## Workflow

### `train` — cross-validation for one model

- Set `mode: train` and choose a model via `training_model`.
- Runs **KFold CV** (5 folds by default) on the full training set.
- Writes metrics to `logs/{experiment_name}.txt`.
- If `save_best_model: True`, updates `logs/{model}.txt` when CV score improves.

Checkpoints are not saved in this mode.

### `fit` — full pipeline (CV → best model → submission)

Default mode. One command runs the entire flow:

1. **CV** for every model in `models_to_evaluate`
2. Optional update of `logs/{model}.txt` when `save_best_model: True`
3. Selection of the best model by CV metric
4. Retrain on **80%** of train (`fit.val_size: 0.2` — 20% for validation)
5. `submission.csv` — predictions on test
6. `result.md` — CV summary table (**Model**, **CV**, **CV STD**)

Supported models: `regression`, `ridge`, `lasso`, `elasticnet`, `KNN`, `decision_tree`, `random_forest`, `catboost`, `lightgbm`, `xgboost`, `voting`, `stacking`, `DNN`.

## Leaderboard submissions

| Model    | CV        | CV STD   | LB      | Date       |
| -------- | --------- | -------- | ------- | ---------- |
| stacking | -0.014590 | 0.002726 | 0.12215 | 2026-06-30 |
| voting   | -0.014546 | 0.002597 | 0.12378 | 2026-06-30 |
| catboost | -0.014496 | 0.002782 | 0.12408 | 2026-06-30 |
| xgboost  | -0.015429 | 0.002542 | 0.12723 | 2026-06-28 |
| lightgbm | -0.015512 | 0.002317 | 0.12997 | 2026-06-28 |
| DNN      | -0.019884 | 0.004727 | 0.14251 | 2026-06-28 |
