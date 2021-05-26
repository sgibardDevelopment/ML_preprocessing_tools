import pandas as pd
from sklearn.model_selection import train_test_split
from deal_with_missing_values import DealWithMissingValues
from deal_with_categorical_variables import DealWithCategoricalVariables
from imputer import Imputer
from sklearn.preprocessing import OneHotEncoder


class Dataset:

    def __init__(self, dataset: pd.DataFrame, target: pd.DataFrame, split=0.8):
        self.__check_split_value_input(split)
        self.__create_training_and_validation_set(dataset, target, split)
        self.dataset = dataset
        self.__saved_dataset = dataset
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

    def reset_dataset(self):
        self.dataset = self.__saved_dataset

    def drop_columns_with_missing_val(self, missing_val_number_limiter=None, level="above"):
        self.dataset = DealWithMissingValues(self.dataset).drop_columns_with_missing_val(missing_val_number_limiter,
                                                                                         level)

    def impute_columns_with_missing_val(self, imputer: Imputer):
        self.dataset = DealWithMissingValues(self.dataset).impute_columns_with_missing_val(imputer)

    def locate_missing_values(self, missing_val_number_limiter=None, level="above"):
        self.dataset = DealWithMissingValues(self.dataset).locate_missing_values(missing_val_number_limiter, level)

    def drop_numerical_columns(self):
        self.dataset = DealWithCategoricalVariables(self.dataset).drop_numerical_columns()

    def drop_categorical_columns(self, unique_var_limiter=None, cardinal_type="high"):
        self.dataset = DealWithCategoricalVariables(self.dataset).drop_categorical_columns(unique_var_limiter,
                                                                                           cardinal_type)

    def apply_one_hot_encoding(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame,
                               unique_var_limiter=None, cardinal_type="high"):
        self.dataset = DealWithCategoricalVariables(self.dataset).apply_one_hot_encoding(one_hot_encoder, training_set,
                                                                                         unique_var_limiter,
                                                                                         cardinal_type)