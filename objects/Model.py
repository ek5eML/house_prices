from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor


MODELS = {
  'regression': lambda params: LinearRegression(**params),
  'ridge': lambda params: Ridge(**params),
  'lasso': lambda params: Lasso(**params),
  'elasticnet': lambda params: ElasticNet(**params),
  'KNN': lambda params: KNeighborsRegressor(**params),
  'decision_tree': lambda params: DecisionTreeRegressor(**params),
  'random_forest': lambda params: RandomForestRegressor(**params),
  'catboost': lambda params: CatBoostRegressor(**params),
  'lightgbm': lambda params: LGBMRegressor(**params),
  'xgboost': lambda params: XGBRegressor(**params),
  'ensemble': None,
}

ENSEMBLE_MODELS = {
  'voting': VotingRegressor,
  'stacking': StackingRegressor,
}


def _parse_ensemble_models(config) -> tuple[list[str], list[float]]:
  names = []
  weights = []
  for model_name, weight in config.models_params.ensemble.models.items():
    weight = float(weight)
    if weight <= 0:
      continue
    names.append(model_name)
    weights.append(weight)

  if not names:
    raise ValueError('Ensemble must include at least one model with weight > 0')

  return names, weights


def __get_model_name(config, name: str = ''):
  model_name = name if name else config.training_model
  
  if model_name not in MODELS:
    raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS)}")

  return model_name

def get_model(config, name: str = '', params: dict | None = None):
  name = __get_model_name(config, name)
  
  params = params if params is not None else config.models_params[name]
  
  if name == 'ensemble':
    selected_models, weights = _parse_ensemble_models(config)
    estimators = [
      (model_name, MODELS[model_name](config.models_params[model_name]))
      for model_name in selected_models
    ]

    ensemble_type = config.models_params.ensemble.type
    ensemble_params = {}
    if ensemble_type == 'voting':
      ensemble_params['weights'] = weights
    elif ensemble_type == 'stacking':
      ensemble_params['final_estimator'] = RidgeCV()

    model = ENSEMBLE_MODELS[ensemble_type](
      estimators=estimators,
      **ensemble_params,
    )
  else:
    model = MODELS[name](params)
  
  return model
