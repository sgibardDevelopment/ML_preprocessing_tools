from dataset import Dataset
import pandas as pd

class Training_set(Dataset):

    def __init__(self, dataset: pd.DataFrame, target: pd.DataFrame, split=0.8):
        super().__init__(dataset, target, split)
        self.dataset = self._training_set
        self.target = self._training_target
