import pandas as pd
from dataframe_creator import Dataframe_Creator
from dataset import Dataset
from sklearn.model_selection import train_test_split
from deal_with_missing_values import DealWithMissingValues
from model_generator import ModelGenerator
from model_evaluator import ModelEvaluator
from imputer import Imputer

dataframe_creator = Dataframe_Creator("train.csv", "test.csv", 'SalePrice')

# Get rid of string columns :
dataframe_creator.clean_dataset_of_string_columns()
dataframe_creator.clean_test_set_of_string_columns()

# Cut the dataset between a training set and a validation set :
dataset = Dataset(dataframe_creator.dataset, dataframe_creator.target, 0.8)

print(dataset.dataset)
#dataset.drop_columns_with_missing_val()
#imputer = Imputer('simple', dataset.training_set)
#dataset.impute_columns_with_missing_val(imputer)
dataset.locate_missing_values()
print(dataset.dataset)

#deal_with_missing_val_for_training_set = DealWithMissingValues(dataset.training_set)
#deal_with_missing_val_for_validation_set = DealWithMissingValues(dataset.validation_set)

#imputer = Imputer('simple', dataset.training_set)
#print(deal_with_missing_val_for_training_set.impute_columns_with_missing_val(imputer))


'''
missing_val_number_limiter = 10
deal_with_missing_val_for_X_train = DealWithMissingValues(X_train, "train", missing_val_number_limiter)
deal_with_missing_val_for_X_valid = DealWithMissingValues(X_valid, "valid", missing_val_number_limiter)
deal_with_missing_val_for_X_test = DealWithMissingValues(X_test, "test", missing_val_number_limiter)

reduced_X_train = deal_with_missing_val_for_X_train.drop_columns_with_missing_val()
reduced_X_valid = deal_with_missing_val_for_X_valid.drop_columns_with_missing_val()
model_drop = ModelGenerator(reduced_X_train, reduced_X_valid, y_train, y_valid)
model_drop.generate_random_forest_regressor_model(n_estimators=100, random_state=0)
model_drop.train()
prediction_drop = model_drop.predict(reduced_X_valid)
model_drop_evaluation = ModelEvaluator(y_valid, prediction_drop)
print(model_drop_evaluation.evaluate_and_get_mean_absolute_error())

imputer = Imputer('simple', X_train)
imputed_X_train = deal_with_missing_val_for_X_train.impute_columns_with_missing_val(imputer, X_train)
imputed_X_valid = deal_with_missing_val_for_X_valid.impute_columns_with_missing_val(imputer, X_valid)
model_imputed = ModelGenerator(imputed_X_train, imputed_X_valid, y_train, y_valid)
model_imputed.generate_random_forest_regressor_model(n_estimators=100, random_state=0)
model_imputed.train()
prediction_imputed = model_imputed.predict(imputed_X_valid)
model_imputed_evaluation = ModelEvaluator(y_valid, prediction_imputed)
print(model_imputed_evaluation.evaluate_and_get_mean_absolute_error())

zero_one_X_train = deal_with_missing_val_for_X_train.replace_missing_val_columns_with_zero_one_columns(imputer)
zero_one_X_valid = deal_with_missing_val_for_X_valid.replace_missing_val_columns_with_zero_one_columns(imputer)
model_zero_one = ModelGenerator(zero_one_X_train, zero_one_X_valid, y_train, y_valid)
model_zero_one.generate_random_forest_regressor_model(n_estimators=100, random_state=0)
model_zero_one.train()
prediction_zero_one = model_zero_one.predict(zero_one_X_valid)
model_zero_one_evaluation = ModelEvaluator(y_valid, prediction_zero_one)
print(model_zero_one_evaluation.evaluate_and_get_mean_absolute_error())

zero_one_X_test = deal_with_missing_val_for_X_test.replace_missing_val_columns_with_zero_one_columns(imputer)
prediction_test = model_zero_one.predict(zero_one_X_test)
model_zero_one.generate_sumbmission_file(X_test, prediction_test)
'''