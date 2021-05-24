import pandas as pd
from sklearn.model_selection import train_test_split
from deal_with_missing_values import DealWithMissingValues
from imputer import Imputer


class Dataset:

    def __init__(self, dataset: pd.DataFrame, target: pd.DataFrame, split=0.8):
        self.__check_split_value_input(split)
        self.__create_training_and_validation_set(dataset, target, split)
        self.dataset = dataset
        self.target = target
        self.split = split

    def __check_split_value_input(self, split: int):
        if split < 0.0:
            raise (ValueError("Error: Dataset - Split value must be positive."))
        if split > 1.0:
            raise (ValueError("Error: Dataset - Split value must be less than 1.0."))

    def __create_training_and_validation_set(self, dataset: pd.DataFrame, target: pd.DataFrame, split: int):
        self._training_set, self._validation_set, self._training_target, self._validation_target = train_test_split(
            dataset,
            target,
            train_size=split,
            test_size=1 - split,
            random_state=0
        )

    def drop_columns_with_missing_val(self, missing_val_number_limiter=None):
        self.dataset = DealWithMissingValues(self.dataset).drop_columns_with_missing_val(missing_val_number_limiter)

    def impute_columns_with_missing_val(self, imputer: Imputer, missing_val_number_limiter=None):
        self.dataset = DealWithMissingValues(self.dataset, missing_val_number_limiter).impute_columns_with_missing_val(imputer)

    def locate_missing_values(self, missing_val_number_limiter=None):
        self.dataset = DealWithMissingValues(self.dataset, missing_val_number_limiter).locate_missing_values()