import pandas as pd
from dataset import Dataset
from deal_with_categorical_variables import DealWithCategoricalVariables
from sklearn.preprocessing import OneHotEncoder

data_full_path_name = "debut.csv"
working_set = pd.read_csv(data_full_path_name, index_col="Id")
target = working_set.Prix
working_set.drop(["Prix"], axis=1, inplace=True)

dataset = Dataset(working_set, target)

deal_with_categorical_variables = DealWithCategoricalVariables(dataset.dataset)
'''dataset.dataset = deal_with_categorical_variables.drop_categorical_columns()'''
print(deal_with_categorical_variables.unique_entries_per_categorical_columns)
print("col_with_categ_data : ", deal_with_categorical_variables.col_with_categ_data)
'''print("high_card_col_with_categ_data : ", deal_with_categorical_variables.high_card_col_with_categ_data)
print("low_card_col_with_categ_data : ", deal_with_categorical_variables.low_card_col_with_categ_data)'''

one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
one_hot_encoder.fit(dataset.dataset)
one_hot_encoder.transform(dataset.dataset)

print(deal_with_categorical_variables.apply_one_hot_encoding(one_hot_encoder, dataset.dataset))

'''print(dataset.target)'''