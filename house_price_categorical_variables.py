import pandas as pd
from dataset import Dataset
from deal_with_categorical_variables import DealWithCategoricalVariables

data_full_path_name = "debut.csv"
working_set = pd.read_csv(data_full_path_name, index_col="Id")
target = working_set.Prix
working_set.drop(["Prix"], axis=1, inplace=True)

dataset = Dataset(working_set, target)

print(dataset.dataset)
deal_with_categorical_variables = DealWithCategoricalVariables(dataset.dataset)
print(deal_with_categorical_variables.drop_categorical_columns())


'''print(dataset.dataset)
print(dataset.target)'''