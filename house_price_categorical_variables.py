import pandas as pd
from dataset import Dataset
from deal_with_categorical_variables import DealWithCategoricalVariables
from sklearn.preprocessing import OneHotEncoder

data_full_path_name = "debut.csv"
working_set = pd.read_csv(data_full_path_name, index_col="Id")
target = working_set.Prix
working_set.drop(["Prix"], axis=1, inplace=True)

dataset = Dataset(working_set, target)

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
one_hot_encoder.fit(dataset.dataset)
one_hot_encoder.transform(dataset.dataset)

dataset.apply_one_hot_encoding(one_hot_encoder, dataset.dataset, unique_var_limiter=7, cardinal_type="low")
print(dataset.dataset)
dataset.reset_dataset()
print(dataset.dataset)

#print(deal_with_categorical_variables.apply_one_hot_encoding(one_hot_encoder, dataset.dataset, unique_var_limiter=7, cardinal_type="high"))

'''print(dataset.target)'''