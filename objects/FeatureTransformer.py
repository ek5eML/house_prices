from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from objects.DataLoader import DataLoader


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

OUTLIERS_IDS = [30, 88, 462, 631, 1322]

class FeatureTransformer(BaseEstimator, TransformerMixin):
  """Prepare raw tabular features and encode categoricals from train+test vocab."""

  def __init__(self, config):
    self.config = config

  def _drop_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
    outlier_ids = df.index.intersection(OUTLIERS_IDS)
    if outlier_ids.empty:
      return df
    
    return df.drop(index=outlier_ids)

  def _prepare(self, df: pd.DataFrame, *, drop_outliers: bool = False) -> pd.DataFrame:
    """Fill missing values, engineer features, drop unused columns."""
    df = df.copy()
    df = self._fill_missing(df)
    df = self._build_features(df)
    df = df.drop(columns=DROP_COLUMNS, errors='ignore')
    if drop_outliers:
      df = self._drop_outliers(df)

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

    df['TotalBath'] = (
      df['FullBath'] + df['BsmtFullBath']
      + 0.5 * (df['HalfBath'] + df['BsmtHalfBath'])
    )

    df['TotalPorchSF'] = (
      df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch']
      + df['ScreenPorch'] + df['WoodDeckSF']
    )

    for new_col, col in NUMERIC_TO_BOOL_FEATURE.items():
      df[new_col] = (df[col] > 0).astype(int)

    for col in ('MSSubClass', 'YrSold', 'MoSold'):
      df[col] = df[col].astype(str)

    return df

  def _build_categorical_vocab(self, df: pd.DataFrame) -> dict[str, list[str]]:
    vocab = {}
    for col in self.categorical_columns_:
      values = df[col].astype(str).fillna('__unknown__')
      vocab[col] = sorted(values.unique().tolist())
    return vocab

  def _build_encoded_column_names(self) -> list[str]:
    names = list(self.numerical_columns_)
    for col in self.categorical_columns_:
      for category in self.categorical_vocab_[col][1:]:
        names.append(f'{col}_{category}')
    return names

  def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals using vocab built on combined train+test data."""
    num_df = df[self.numerical_columns_].fillna(self.numeric_fill_values_)
    if not self.categorical_columns_:
      return num_df

    encoded_parts = [num_df]
    for col in self.categorical_columns_:
      values = df[col].astype(str).fillna('__unknown__')
      for category in self.categorical_vocab_[col][1:]:
        encoded_parts.append(
          pd.Series(
            (values == category).astype(int),
            index=df.index,
            name=f'{col}_{category}',
          )
        )

    return pd.concat(encoded_parts, axis=1)

  def fit(self, X=None, y=None):
    """Learn fill values and category vocab from combined train and test features."""
    data_loader = DataLoader(self.config)
    combined_X = data_loader.get_combined_features()

    self.mode_fill_values_ = {}
    for col in MODE_FILL_COLUMNS:
      if col not in combined_X.columns:
        continue
      mode = combined_X[col].dropna().mode()
      if not mode.empty:
        self.mode_fill_values_[col] = mode.iloc[0]

    df = self._prepare(combined_X, drop_outliers=True)
    self.categorical_columns_ = (
      df.select_dtypes(include=['object', 'category']).columns.tolist()
    )
    self.numerical_columns_ = df.select_dtypes(include='number').columns.tolist()
    self.numeric_fill_values_ = df[self.numerical_columns_].median()
    self.categorical_vocab_ = self._build_categorical_vocab(df)
    self.encoded_columns_ = self._build_encoded_column_names()

    return self

  def transform(self, X):
    """Transform raw features into numeric matrix with fixed column order."""
    df = self._prepare(X)
   
    return self._encode(df)
