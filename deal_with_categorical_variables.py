import pandas as pd


class DealWithCategoricalVariables:

    def __init__(self, working_set: pd.DataFrame, unique_var_limiter=None):
        self.working_set = working_set
        self.unique_var_limiter = unique_var_limiter

        if unique_var_limiter is None:
            self.__create_list_of_cols_with_categ_data()
        else:
            self.__create_list_of_cols_with_categ_data_according_to_limiter()

    def __create_list_of_cols_with_categ_data(self):
        self.col_with_categ_data = [col for col in self.working_set.columns if self.working_set[col].dtypes == 'object']

    def __create_list_of_cols_with_categ_data_according_to_limiter(self):
        self.col_with_categ_data = [col for col in self.working_set.columns if
                                    len(self.working_set[col].unique()) > self.unique_var_limiter]

    def drop_numerical_columns(self):
        numerical_columns = [col for col in self.working_set.columns if self.working_set[col].dtypes == int]
        return self.working_set.drop(numerical_columns, axis=1)


