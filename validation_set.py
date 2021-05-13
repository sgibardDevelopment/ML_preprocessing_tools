from dataset import Dataset
from model_evaluator import ModelEvaluator
import pandas as pd
import numpy as np

class Validation_set(Dataset):

    def __init__(self, dataset: pd.DataFrame, target: pd.DataFrame, split=0.8):
        super().__init__(dataset, target, split)
        self.dataset = self._validation_set
        self.target = self._validation_target

    def evaluate_model_with_mean_absolute_error(self, prediction: np.ndarray):
        return ModelEvaluator(self.dataset, prediction).evaluate_mean_absolute_error()