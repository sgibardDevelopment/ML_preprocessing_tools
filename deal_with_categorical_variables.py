import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DealWithCategoricalVariables:

    def __init__(self, working_set: pd.DataFrame, unique_var_limiter=None):
        self.working_set = working_set
        self.unique_var_limiter = unique_var_limiter
        self.__get_unique_entries_per_categorical_columns()

        if unique_var_limiter is None:
            self.__get_categorical_columns()
        else:
            self.__get_categorical_columns_according_to_limiter()

    def __get_categorical_columns(self):
        self.col_with_categ_data = [col for col in self.working_set.columns if self.working_set[col].dtypes == 'object']

    def __get_categorical_columns_according_to_limiter(self):
        self.col_with_categ_data = [col for col in self.working_set.columns if
                                    len(self.working_set[col].unique()) > self.unique_var_limiter]

    def drop_numerical_columns(self):
        numerical_columns = [col for col in self.working_set.columns if self.working_set[col].dtypes == int]
        return self.working_set.drop(numerical_columns, axis=1)

    def drop_categorical_columns(self):
        return self.working_set.drop(self.col_with_categ_data, axis=1)

    def __get_unique_entries_per_categorical_columns(self):
        self.unique_entries_per_categorical_columns = dict()
        for col in self.working_set.columns:
            self.unique_entries_per_categorical_columns[col] = len(self.working_set[col].unique())

    def apply_one_hot_encoding(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame):
        categorical_var_working_set = self.drop_numerical_columns()
        categorical_var_training_set = self.__create_categorical_var_training_set(training_set)
        numerical_working_set = self.drop_categorical_columns()
        one_hot_encoded_working_set = self.__one_hot_encode(one_hot_encoder, categorical_var_working_set, categorical_var_training_set)
        one_hot_encoded_working_set = self.__reformat_one_hot_encoded_array_into_df(one_hot_encoded_working_set)
        return pd.concat([numerical_working_set, one_hot_encoded_working_set], axis=1)

    def __create_categorical_var_training_set(self, training_set: pd.DataFrame):
        deal_with_categorical_var_for_training_set = DealWithCategoricalVariables(training_set)
        return deal_with_categorical_var_for_training_set.drop_numerical_columns()

    def __one_hot_encode(self, one_hot_encoder: OneHotEncoder, categorical_var_working_set: pd.DataFrame, categorical_var_training_set: pd.DataFrame):
        one_hot_encoder.fit(categorical_var_training_set)
        return one_hot_encoder.transform(categorical_var_working_set)

    def __reformat_one_hot_encoded_array_into_df(self, one_hot_encoded_array: np.ndarray):
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array)
        one_hot_encoded_df.index = self.working_set.index
        return one_hot_encoded_df

