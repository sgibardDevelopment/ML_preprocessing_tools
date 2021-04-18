from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class ModelGenerator:

    def __init__(self, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.DataFrame, y_valid: pd.DataFrame):
        self._X_train = X_train
        self._X_valid = X_valid
        self._y_train = y_train
        self._y_valid = y_valid
        self._model_is_created = False
        self._model_is_trained = False
        self._prediction_is_done = False


    def generate_random_forest_regressor_model(self, n_estimators:int, random_state:int):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self._model_is_created = True

    def train(self):
        if self._model_is_created:
            self.model.fit(self._X_train, self._y_train)
            self._model_is_trained = True
        else:
            print("You need to create your model.")

    def predict(self, X: pd.DataFrame):
        if self._model_is_trained:
            self._prediction_is_done = True
            return self.model.predict(X)
        else:
            print("You need to train your model before prediction.")

    def generate_sumbmission_file(self, X_test, prediction_test):
        output = pd.DataFrame({'Id': X_test.index,
                               'SalePrice': prediction_test})
        output.to_csv('submission.csv', index=False)
