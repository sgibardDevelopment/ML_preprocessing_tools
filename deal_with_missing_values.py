import pandas as pd
from imputer import Imputer

class DealWithMissingValues:

    def __init__(self, working_set: pd.DataFrame):
        self.working_set = working_set

    def drop_columns_with_missing_val(self, missing_val_number_limiter=None, level="above"):
        self.__initialize_list_of_cols_with_missing_values(missing_val_number_limiter, level)
        return self.working_set.drop(self.cols_with_missing_values, axis=1)

    def impute_columns_with_missing_val(self, imputer: Imputer):
        return imputer.transform_with_imputer(self.working_set)

    def locate_missing_values(self, missing_val_number_limiter=None, level="above"):
        self.__initialize_list_of_cols_with_missing_values(missing_val_number_limiter, level)
        for col in self.cols_with_missing_values:
            self.working_set[col + " has na ?"] = self.working_set[col].isnull().astype(int)
        return self.working_set

    def __initialize_list_of_cols_with_missing_values(self, missing_val_number_limiter, level):
        if missing_val_number_limiter is None:
            self.__create_list_of_cols_with_missing_values()
        else:
            self.__create_list_of_cols_with_missing_values_according_to_limiter(missing_val_number_limiter, level)
        self.__check_if_cols_with_missing_values_is_empty()

    def __create_list_of_cols_with_missing_values(self):
        self.cols_with_missing_values = [col for col in self.working_set.columns if self.working_set[col].isnull().any()]

    def __create_list_of_cols_with_missing_values_according_to_limiter(self, missing_val_number_limiter, level):
        missing_val_count_by_column = (self.working_set.isnull().sum())
        if level is "above":
            rule = (missing_val_count_by_column >= missing_val_number_limiter)
            self.cols_with_missing_values = missing_val_count_by_column[rule].index.values.tolist()
        elif level is "below":
            rule = (missing_val_count_by_column <= missing_val_number_limiter)
            self.cols_with_missing_values = missing_val_count_by_column[rule].index.values.tolist()
        else:
            raise (ValueError("Error: You must select 'above' or 'below' limiter"))

    def __check_if_cols_with_missing_values_is_empty(self):
        if len(self.cols_with_missing_values) is 0:
            print("There are no missing values in dataset according to limiter.")






