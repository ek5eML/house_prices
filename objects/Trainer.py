def get_runner(config):
  if config.training_model == 'DNN':
    from objects.dnn.runner import DNNRunner
    return DNNRunner(config)

  from objects.sklearn.runner import SklearnRunner
  return SklearnRunner(config)


class Trainer:
  def __init__(self, config):
    self.config = config
    self.runner = get_runner(config)

  def run_cv(self, name: str = '', params: dict | None = None):
    return self.runner.run_cv(name, params)

  def fit_full(self, name: str = '', params: dict | None = None):
    return self.runner.fit_full(name, params)

  def predict(self, test_data):
    return self.runner.predict(test_data)
