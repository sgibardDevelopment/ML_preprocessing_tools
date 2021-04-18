from sklearn.impute import SimpleImputer
import pandas as pd

class Imputer:

    def __init__(self, imputer_type: str, X_train: pd.DataFrame):
        self.imputer_type = imputer_type
        self.X_train = X_train

        if imputer_type == "simple":
            self.my_imputer = self.__train_simple_imputer()

    def __train_simple_imputer(self):
        my_simple_imputer = SimpleImputer()
        my_simple_imputer.fit(self.X_train)
        return my_simple_imputer

    def transform_with_imputer(self, X: pd.DataFrame):
        imputed_X = pd.DataFrame(self.my_imputer.transform(X))
        imputed_X.columns = X.columns
        return imputed_X



