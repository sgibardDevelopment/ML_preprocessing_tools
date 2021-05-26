import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DealWithCategoricalVariables:

    def __init__(self, working_set: pd.DataFrame):
        self.working_set = working_set
        self.__get_categorical_columns()
        self.__get_unique_entries_per_categorical_columns()

    def __get_unique_entries_per_categorical_columns(self):
        self.unique_entries_per_categorical_columns = dict()
        for col in self.col_with_categ_data:
            self.unique_entries_per_categorical_columns[col] = len(self.working_set[col].unique())

    def __get_categorical_columns(self):
        self.col_with_categ_data = [col for col in self.working_set.columns if self.working_set[col].dtype == "object"]

    def __create_high_and_low_cardinality_parameters_according_to_limiter(self, unique_var_limiter: int):
        if unique_var_limiter is not None:
            self.__get_high_cardinality_categorical_columns_according_to_limiter(unique_var_limiter)
            self.__get_low_cardinality_categorical_columns_according_to_limiter(unique_var_limiter)

    def __get_high_cardinality_categorical_columns_according_to_limiter(self, unique_var_limiter: int):
        self.__high_card_col_with_categ_data = [col for col in self.col_with_categ_data if
                                                self.unique_entries_per_categorical_columns[col] >= unique_var_limiter]

    def __get_low_cardinality_categorical_columns_according_to_limiter(self, unique_var_limiter: int):
        self.__low_card_col_with_categ_data = [col for col in self.col_with_categ_data if
                                               self.unique_entries_per_categorical_columns[col] <= unique_var_limiter]

    def drop_numerical_columns(self):
        numerical_columns = [col for col in self.working_set.columns if self.working_set[col].dtypes == int]
        return self.working_set.drop(numerical_columns, axis=1)

    def drop_categorical_columns(self, unique_var_limiter=None, cardinal_type="high"):
        if unique_var_limiter is None:
            return self.working_set.drop(self.col_with_categ_data, axis=1)
        else:
            self.__create_high_and_low_cardinality_parameters_according_to_limiter(unique_var_limiter)
            return self.__select_dropping_according_to_cardinal_type(cardinal_type)

    def __select_dropping_according_to_cardinal_type(self, cardinal_type: str):
        if cardinal_type is "high":
            return self.__drop_high_card()
        elif cardinal_type is "low":
            return self.__drop_low_card()
        else:
            raise (ValueError("Error: You must select 'high' or 'low' cardinality"))

    def __drop_high_card(self):
        return self.working_set.drop(self.__high_card_col_with_categ_data, axis=1)

    def __drop_low_card(self):
        return self.working_set.drop(self.__low_card_col_with_categ_data, axis=1)

    def apply_one_hot_encoding(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame,
                               unique_var_limiter=None, cardinal_type="high"):
        self.__create_high_and_low_cardinality_parameters_according_to_limiter(unique_var_limiter)
        one_hot_encoded_working_set = self.__one_hot_encode_working_set(one_hot_encoder, training_set,
                                                                        unique_var_limiter, cardinal_type)
        one_hot_encoded_working_set = self.__reformat_one_hot_encoded_array_into_df(one_hot_encoded_working_set)
        not_oh_encoded_working_set = self.__retrieve_not_oh_encoded_working_set(unique_var_limiter, cardinal_type)
        return pd.concat([not_oh_encoded_working_set, one_hot_encoded_working_set], axis=1)

    def __one_hot_encode_working_set(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame,
                                     unique_var_limiter: int, cardinal_type: str):
        if unique_var_limiter is None:
            return self.__one_hot_encode(one_hot_encoder, self.working_set[self.col_with_categ_data],
                                         training_set[self.col_with_categ_data])
        else:
            return self.__select_oh_encoding_according_to_cardinal_type(one_hot_encoder, training_set, cardinal_type)

    def __select_oh_encoding_according_to_cardinal_type(self, one_hot_encoder: OneHotEncoder, training_set:pd.DataFrame, cardinal_type: str):
        if cardinal_type is "high":
            return self.__oh_encode_high_card(one_hot_encoder, training_set)
        elif cardinal_type is "low":
            return self.__oh_encode_low_card(one_hot_encoder, training_set)
        else:
            raise (ValueError("Error: You must select 'high' or 'low' cardinality"))

    def __oh_encode_high_card(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame):
        return self.__one_hot_encode(one_hot_encoder, self.working_set[self.__high_card_col_with_categ_data],
                                     training_set[self.__high_card_col_with_categ_data])

    def __oh_encode_low_card(self, one_hot_encoder: OneHotEncoder, training_set: pd.DataFrame):
        return self.__one_hot_encode(one_hot_encoder, self.working_set[self.__low_card_col_with_categ_data],
                                     training_set[self.__low_card_col_with_categ_data])

    def __one_hot_encode(self, one_hot_encoder: OneHotEncoder, categorical_var_working_set: pd.DataFrame,
                         categorical_var_training_set: pd.DataFrame):
        one_hot_encoder.fit(categorical_var_training_set)
        return one_hot_encoder.transform(categorical_var_working_set)

    def __reformat_one_hot_encoded_array_into_df(self, one_hot_encoded_array: np.ndarray):
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded_array)
        one_hot_encoded_df.index = self.working_set.index
        return one_hot_encoded_df

    def __retrieve_not_oh_encoded_working_set(self, unique_var_limiter: int, cardinal_type: str):
        deal_with_categorical_var_for_working_set = DealWithCategoricalVariables(self.working_set)
        return deal_with_categorical_var_for_working_set.drop_categorical_columns(unique_var_limiter, cardinal_type)
