from dataset import Dataset
import pandas as pd


class Test_set(Dataset):

    def __init__(self, testset: pd.DataFrame):
        self.dataset = testset
