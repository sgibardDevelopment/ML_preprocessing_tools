from sklearn.impute import SimpleImputer
import pandas as pd


class Imputer:

    def __init__(self, imputer_type: str, imputer_training_set: pd.DataFrame):
        self.__training_set = imputer_training_set

        if imputer_type == "simple":
            self.__my_imputer = self.__train_simple_imputer()

    def __train_simple_imputer(self):
        my_simple_imputer = SimpleImputer()
        my_simple_imputer.fit(self.__training_set)
        return my_simple_imputer

    def transform_with_imputer(self, working_set: pd.DataFrame):
        imputed_set = pd.DataFrame(self.__my_imputer.transform(working_set))
        imputed_set.columns = working_set.columns
        return imputed_set
