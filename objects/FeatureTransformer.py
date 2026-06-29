from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


CATEGORICAL_COLUMNS = (
  'MSZoning',
  'Street',
  'Alley',
  'LotShape',
  'LandContour',
  'Utilities',
  'LotConfig',
  'LandSlope',
  'Neighborhood',
  'Condition1',
  'Condition2',
  'BldgType',
  'HouseStyle',
  'RoofStyle',
  'RoofMatl',
  'Exterior1st',
  'Exterior2nd',
  'MasVnrType',
  'ExterQual',
  'ExterCond',
  'Foundation',
  'BsmtQual',
  'BsmtCond',
  'BsmtExposure',
  'BsmtFinType1',
  'BsmtFinType2',
  'Heating',
  'HeatingQC',
  'CentralAir',
  'Electrical',
  'KitchenQual',
  'Functional',
  'FireplaceQu',
  'GarageType',
  'GarageFinish',
  'GarageQual',
  'GarageCond',
  'PavedDrive',
  'PoolQC',
  'Fence',
  'MiscFeature',
  'SaleType',
  'SaleCondition',
)

NUMERICAL_COLUMNS = (
  'MSSubClass',
  'LotFrontage',
  'LotArea',
  'OverallQual',
  'OverallCond',
  'YearBuilt',
  'YearRemodAdd',
  'MasVnrArea',
  'BsmtFinSF1',
  'BsmtFinSF2',
  'BsmtUnfSF',
  'TotalBsmtSF',
  '1stFlrSF',
  '2ndFlrSF',
  'LowQualFinSF',
  'GrLivArea',
  'BsmtFullBath',
  'BsmtHalfBath',
  'FullBath',
  'HalfBath',
  'BedroomAbvGr',
  'KitchenAbvGr',
  'TotRmsAbvGrd',
  'Fireplaces',
  'GarageYrBlt',
  'GarageCars',
  'GarageArea',
  'WoodDeckSF',
  'OpenPorchSF',
  'EnclosedPorch',
  '3SsnPorch',
  'ScreenPorch',
  'PoolArea',
  'MiscVal',
  'MoSold',
  'YrSold',
)

DROP_COLUMNS = (
  'PoolQC',
  'MiscFeature',
  'Alley',
  'Fence',
  'LotFrontage',
  'MasVnrArea',
  'Utilities',
  'Street',
)

ZERO_FILL_COLUMNS = ('GarageYrBlt', 'GarageArea', 'GarageCars')

NONE_FILL_COLUMNS = (
  'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
  'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
  'MasVnrType', 'FireplaceQu',
)

MODE_FILL_COLUMNS = (
  'Functional', 'Electrical', 'KitchenQual',
  'Exterior1st', 'Exterior2nd', 'SaleType',
)

NUMERIC_TO_BOOL_FEATURE = {
  'hasPool': 'PoolArea',
  'has2ndFloor': '2ndFlrSF',
  'hasGarage': 'GarageArea',
  'hasBasement': 'TotalBsmtSF',
  'hasFireplace': 'Fireplaces',
  'hasPorch': 'TotalPorchSF',
}

OUTLIERS_IDS = [ 30, 88, 462, 631, 1322 ]

class FeatureTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, config):
    self.config = config

  def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = self._fill_missing(df)
    df = self._build_features(df)
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')

    # return df.drop(index=OUTLIERS_IDS, errors='ignore')
    return df

  def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
    for col in ZERO_FILL_COLUMNS:
        df[col] = df[col].fillna(0)

    for col in NONE_FILL_COLUMNS:
        df[col] = df[col].fillna('None')

    for col, value in self.mode_fill_values_.items():
        df[col] = df[col].fillna(value)

    return df

  def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:    
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
    df['YrBltAndRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    
    df['TotalBath'] = df['FullBath'] + df['BsmtFullBath'] + 0.5 * (df['HalfBath'] + df['BsmtHalfBath'])
    
    df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch']
      + df['ScreenPorch'] + df['WoodDeckSF'])
    
    for new_col, col in NUMERIC_TO_BOOL_FEATURE.items():
      df[new_col] = (df[col] > 0).astype(int)
    
    for col in ('MSSubClass', 'YrSold', 'MoSold'):
      df[col] = df[col].astype(str)

    return df

  def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
    num_df = df[self.numerical_columns_].fillna(self.numeric_fill_values_)
    if not self.categorical_columns_:
      return num_df

    cat_df = (
      df[self.categorical_columns_]
      .astype(str)
      .fillna('__unknown__')
    )
    encoded = self.encoder_.transform(cat_df)
    encoded_df = pd.DataFrame(
      encoded,
      columns=self.encoder_.get_feature_names_out(self.categorical_columns_),
      index=df.index,
    )

    return pd.concat([num_df, encoded_df], axis=1)

  def fit(self, X: pd.DataFrame, y = None):
    self.mode_fill_values_ = {}
    for col in MODE_FILL_COLUMNS:
      if col not in X.columns:
        continue
      mode = X[col].dropna().mode()
      if not mode.empty:
        self.mode_fill_values_[col] = mode.iloc[0]

    df = self._prepare(X)
    self.categorical_columns_ = (
      df.select_dtypes(include=['object', 'category']).columns.tolist()
    )
    self.numerical_columns_ = df.select_dtypes(include='number').columns.tolist()
    self.numeric_fill_values_ = df[self.numerical_columns_].median()

    if self.categorical_columns_:
      cat_df = (
        df[self.categorical_columns_]
        .astype(str)
        .fillna('__unknown__')
      )
      self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
      self.encoder_.fit(cat_df)
    else:
      self.encoder_ = None

    return self

  def transform(self, X):
    df = self._prepare(X)
    return self._encode(df)