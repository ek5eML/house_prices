import pandas as pd


class DataLoader:
  def __init__(self, config):
    self.config = config

  def load_data(self, path: str) -> pd.DataFrame:
    if self.config.data.id_col:
      df = pd.read_csv(path, index_col=self.config.data.id_col)
    else:
      df = pd.read_csv(path)

    return df

  def load_train(self) -> pd.DataFrame:
    return self.load_data(self.config.paths.path_to_train_data)

  def load_test(self) -> pd.DataFrame:
    return self.load_data(self.config.paths.path_to_test_data)

  def split_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[self.config.data.target_col])
    y = df[self.config.data.target_col]

    return X, y

  def get_combined_X(self) -> tuple[pd.DataFrame, int, pd.Series]:
    train_df = self.load_train()
    y = train_df[self.config.data.target_col]
    X_train = train_df.drop(columns=[self.config.data.target_col])
    X_test = self.load_test()
    n_train = len(X_train)
    X_all = pd.concat([X_train, X_test], axis=0)
    return X_all, n_train, y

  @staticmethod
  def split_combined(X_all: pd.DataFrame, n_train: int):
    return X_all.iloc[:n_train], X_all.iloc[n_train:]
