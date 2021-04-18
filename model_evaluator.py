import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

class ModelEvaluator:

    def __init__(self, y_valid: pd.DataFrame, prediction: np.ndarray):
        self.y_valid = y_valid
        self.prediction = prediction

    def evaluate_and_get_mean_absolute_error(self):
        return mean_absolute_error(self.y_valid, self.prediction)