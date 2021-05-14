import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error


class ModelEvaluator:

    def __init__(self, validation_set: pd.DataFrame, prediction: np.ndarray):
        self.__validation_set = validation_set
        self.__prediction = prediction

    def evaluate_mean_absolute_error(self):
        return mean_absolute_error(self.__validation_set, self.__prediction)
