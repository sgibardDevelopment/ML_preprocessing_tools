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
training_set = Training_set(dataset)
validation_set = Validation_set(dataset)

print(validation_set.dataset)

imputer = Imputer("simple", training_set.dataset)
validation_set.impute_columns_with_missing_val(imputer)

print(validation_set.dataset)


