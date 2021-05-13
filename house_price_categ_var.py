from dataset_creator import Dataset_creator
from deal_with_categ_var import DealWithCategVar
from deal_with_missing_values import DealWithMissingValues
from imputer import Imputer

dataframe_creator = Dataset_creator("train.csv", "test.csv", 'SalePrice')

# Extract the dataset into X_full, the target y and the test set X_test_full :
X_full = dataframe_creator.get_dataset_full()
y = dataframe_creator.get_dataset_target()
X_test_full = dataframe_creator.get_dataset_test_full()

deal_with_categ_var = DealWithCategVar(X_full, 0)
deal_with_categ_var.create_list_of_cols_with_categ_data_according_to_limiter()
'''
deal_with_missing_val_for_X = DealWithMissingValues(X, 'train', 0)
X = deal_with_missing_val_for_X.drop_columns_with_missing_val_according_to_limiter()
print(deal_with_missing_val_for_X.cols_with_missing_values_X)
'''