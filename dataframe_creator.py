import pandas as pd

class Dataframe_Creator:

    def __init__(self, data_full_path_name:str, test_full_path_name:str, target_column_name:str):
        self.X_full = pd.read_csv(data_full_path_name, index_col="Id")
        self.X_test_full = pd.read_csv(test_full_path_name, index_col="Id")
        self.__generate_target_y(target_column_name)
        self.__delete_target_column(target_column_name)

    def __generate_target_y(self, target_column_name:str):
        self.X_full.dropna(axis=0, subset=[target_column_name], inplace=True)
        self.y = self.X_full[target_column_name]

    def __delete_target_column(self, target_column_name:str):
        self.X_full.drop([target_column_name], axis=1, inplace=True)

    def get_df_without_columns_with_string(self):
        X = self.X_full.select_dtypes(exclude=['object'])
        X_test = self.X_test_full.select_dtypes(exclude=['object'])
        return X, X_test

    def get_dataset_full(self):
        return self.X_full

    def get_dataset_target(self):
        return self.y

    def get_dataset_test_full(self):
        return self.X_test_full




