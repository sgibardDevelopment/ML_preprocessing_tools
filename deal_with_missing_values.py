import pandas as pd
from imputer import Imputer

class DealWithMissingValues:

    def __init__(self, X, type: str, missing_val_number_limiter: int):
        self.X = X
        self.type = type
        self.missing_val_number_limiter = missing_val_number_limiter
        self.cols_with_missing_values_X = []

    def __create_list_of_cols_with_missing_values(self):
        self.cols_with_missing_values_X = [col for col in self.X.columns if self.X[col].isnull().any()]

    def __create_list_of_cols_with_missing_values_according_to_limiter(self):
        missing_val_count_by_column_X = (self.X.isnull().sum())
        self.cols_with_missing_values_X = missing_val_count_by_column_X[missing_val_count_by_column_X > self.missing_val_number_limiter].index.values.tolist()
        self.missing_val_count_by_column_X = missing_val_count_by_column_X

    def drop_columns_with_missing_val(self):
        self.__create_list_of_cols_with_missing_values()
        reduced_X = self.X.drop(self.cols_with_missing_values_X, axis=1)
        return reduced_X

    def drop_columns_with_missing_val_according_to_limiter(self):
        self.__create_list_of_cols_with_missing_values_according_to_limiter()
        reduced_X = self.X.drop(self.cols_with_missing_values_X, axis=1)
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
        zero_one_X = self.X
        self.__create_list_of_cols_with_missing_values_according_to_limiter()
        zero_one_X = self.__zero_one_columns_generator(zero_one_X)
        zero_one_X = self.__apply_simple_impute_for_columns_under_limiter(imputer, zero_one_X)
        return zero_one_X

    def __zero_one_columns_generator(self, zero_one_X):
        for col in self.cols_with_missing_values_X:
            zero_one_X[col] = self.X[col].isnull().astype(int)
        return zero_one_X

    def __apply_simple_impute_for_columns_under_limiter(self, imputer: Imputer, X: pd.DataFrame):
        if self.missing_val_number_limiter > 0:
            return self.impute_columns_with_missing_val(imputer, X)






