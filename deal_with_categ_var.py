import pandas as pd


class DealWithCategVar:

    def __init__(self, X: pd.DataFrame, unique_var_limiter=None):
        self.X = X
        self.unique_var_limiter = unique_var_limiter

        if unique_var_limiter == None:
            self.__create_list_of_cols_with_categ_data()
        else:
            self.__create_list_of_cols_with_categ_data_according_to_limiter()

    def __create_list_of_cols_with_categ_data(self):
        self.col_with_categ_data = [col for col in self.X.columns if self.X[col].dtypes == 'object']

    def __create_list_of_cols_with_categ_data_according_to_limiter(self):
        self.col_with_categ_data = [col for col in self.X.columns if
                                    len(self.X[col].unique()) > self.unique_var_limiter]
