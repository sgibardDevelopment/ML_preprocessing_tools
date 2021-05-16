import pandas as pd


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


