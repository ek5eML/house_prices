import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.compose import TransformedTargetRegressor

from objects.FeatureTransformer import FeatureTransformer
from objects.Model import get_model
from utils import inverse_transform_target, transform_target


def build_pipeline(config, name: str = '', params: dict | None = None):
  steps = [
    ('feature_transformer', FeatureTransformer(config)),
  ]

  if config.need_scaler:
    steps.append(('scaler', RobustScaler()))

  steps.append(('model', get_model(config, name, params)))
  
  pipeline = Pipeline(steps=steps)
  
  if config.data.transform_target:
    pipeline = TransformedTargetRegressor(
      regressor=pipeline,
      func=transform_target,
      inverse_func=inverse_transform_target,
    )

  return pipeline
