from dataset import Dataset
import pandas as pd


class Test_set(Dataset):

    def __init__(self, test_set: pd.DataFrame):
        self.dataset = test_set

    def generate_submission_file(self, prediction_test):
        output = pd.DataFrame({'Id': self.dataset.index,
                               'SalePrice': prediction_test})
        output.to_csv('submission.csv', index=False)
