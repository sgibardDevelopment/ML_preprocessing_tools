import pandas as pd
from dataset import Dataset
from training_set import Training_set
from validation_set import Validation_set
from imputer import Imputer

data_full_path_name = "debut_missing_val.csv"
working_set = pd.read_csv(data_full_path_name, index_col="Id")
target = working_set.Money
working_set.drop(["Money"], axis=1, inplace=True)

dataset = Dataset(working_set, target)
#dataset.drop_columns_with_missing_val(missing_val_number_limiter=2, level="below")
dataset.locate_missing_values(missing_val_number_limiter=3, level="below")
dataset.drop_columns_with_missing_val(missing_val_number_limiter=3, level="below")
print(dataset.dataset)


