import pandas as pd
from imputer import Imputer
from dataset import Dataset

class DealWithMissingValues:

    def __init__(self, dataset: pd.DataFrame, missing_val_number_limiter=None):
        self.dataset = dataset
        if missing_val_number_limiter is None:
            self.__create_list_of_cols_with_missing_values()
        else:
            self.__create_list_of_cols_with_missing_values_according_to_limiter(missing_val_number_limiter)

    def __create_list_of_cols_with_missing_values(self):
        self.cols_with_missing_values = [col for col in self.dataset.columns if self.dataset[col].isnull().any()]
        self.__check_if_cols_with_missing_values_is_empty()

    def __create_list_of_cols_with_missing_values_according_to_limiter(self, missing_val_number_limiter: int):
        missing_val_count_by_column = (self.dataset.isnull().sum())
        self.cols_with_missing_values = missing_val_count_by_column[missing_val_count_by_column > missing_val_number_limiter].index.values.tolist()
        self.__check_if_cols_with_missing_values_is_empty()

    def __check_if_cols_with_missing_values_is_empty(self):
        if len(self.cols_with_missing_values) is 0:
            print("There are no missing values in dataset.")

    def drop_columns_with_missing_val(self):
        return self.dataset.drop(self.cols_with_missing_values, axis=1)
'''
    def drop_columns_with_missing_val_according_to_limiter(self):
        self.__create_list_of_cols_with_missing_values_according_to_limiter()
        reduced_X = self.dataset.drop(self.cols_with_missing_values, axis=1)
        return reduced_X

    def impute_columns_with_missing_val(self, imputer: Imputer, X: pd.DataFrame):
        imputed_X = imputer.transform_with_imputer(X)
        return imputed_X

    def replace_missing_val_columns_with_zero_one_columns(self, imputer: Imputer):
        """
        columns with missing values are switched with zero and one columns - for one column :
        0 : when there is a value
        1 : when there is a missing value
        """
        pd.options.mode.chained_assignment = None  # default='warn'
        zero_one_X = self.dataset
        self.__create_list_of_cols_with_missing_values_according_to_limiter()
        zero_one_X = self.__zero_one_columns_generator(zero_one_X)
        zero_one_X = self.__apply_simple_impute_for_columns_under_limiter(imputer, zero_one_X)
        return zero_one_X

    def __zero_one_columns_generator(self, zero_one_X):
        for col in self.cols_with_missing_values:
            zero_one_X[col] = self.dataset[col].isnull().astype(int)
        return zero_one_X

    def __apply_simple_impute_for_columns_under_limiter(self, imputer: Imputer, X: pd.DataFrame):
        if self.missing_val_number_limiter > 0:
            return self.impute_columns_with_missing_val(imputer, X)
            '''






