import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, dataset: pd.DataFrame, target: pd.DataFrame, split=0.8):
        self.__check_split_value_input(split)
        self.__create_training_and_validation_set(dataset, target, split)

    def __check_split_value_input(self, split: int):
        if split < 0.0:
            raise(ValueError("Error: Dataset - Split value must be positive."))
        if split > 1.0:
            raise(ValueError("Error: Dataset - Split value must be less than 1.0."))

    def __create_training_and_validation_set(self, dataset: pd.DataFrame, target: pd.DataFrame, split: int):
        self.training_set, self.validation_set, self.training_target, self.validation_target = train_test_split(
            dataset,
            target,
            train_size=split,
            test_size=1-split,
            random_state=0
        )