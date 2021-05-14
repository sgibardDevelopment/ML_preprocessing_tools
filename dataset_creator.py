import pandas as pd
from dataset import Dataset
from test_set import Test_set


class Dataset_creator:

    def __init__(self, data_full_path_name: str, test_full_path_name: str, target_column_name: str):
        self.__dataset = self.__generate_dataset(data_full_path_name, target_column_name)
        self.__test_set = self.__generate_test_set(test_full_path_name)
        self.__target = self.__generate_target_column(target_column_name)

    def __generate_dataset(self, data_full_path_name: str, target_column_name: str):
        dataset = pd.read_csv(data_full_path_name, index_col="Id")
        self.__drop_rows_where_target_is_na(dataset, target_column_name)
        return dataset

    def __generate_test_set(self, test_full_path_name: str):
        return pd.read_csv(test_full_path_name, index_col="Id")

    def __drop_rows_where_target_is_na(self, set: pd.DataFrame, target_column_name: str):
        set.dropna(axis=0, subset=[target_column_name], inplace=True)

    def __generate_target_column(self, target_column_name: str):
        target = self.__dataset[target_column_name]
        self.__delete_target_column(target_column_name)
        return target

    def __delete_target_column(self, target_column_name: str):
        self.__dataset.drop([target_column_name], axis=1, inplace=True)

    def create_dataset(self, split=0.8):
        return Dataset(self.__dataset, self.__target, split)

    def create_test_set(self):
        return Test_set(self.__test_set)

    def clean_dataset_of_string_columns(self):
        self.__dataset = self.__dataset.select_dtypes(exclude=['object'])

    def clean_test_set_of_string_columns(self):
        self.__test_set = self.__test_set.select_dtypes(exclude=['object'])
