from dataset_creator import Dataset_creator
from training_set import Training_set
from validation_set import Validation_set
from sklearn.ensemble import RandomForestRegressor
from imputer import Imputer

dataset_creator = Dataset_creator("train.csv", "test.csv", "SalePrice")

# Get rid of string columns :
dataset_creator.clean_dataset_of_string_columns()
dataset_creator.clean_test_set_of_string_columns()

# Create the dataset that will generate training/validation_set with a specific split :
dataset = dataset_creator.create_dataset(split=0.8)

# Drop columns with missing values method
# - Cut the dataset between a training set and a validation set :
reduced_training_set = Training_set(dataset)
reduced_validation_set = Validation_set(dataset)

# - reduce : columns with missing val are dropped :
reduced_training_set.drop_columns_with_missing_val()
reduced_validation_set.drop_columns_with_missing_val()
'''
# - Use random forest regressor :
model = RandomForestRegressor(n_estimators=100, random_state=0)

# - Let's train the model :
model.fit(reduced_training_set.dataset, reduced_training_set.target)

# - Apply prediction :
prediction = model.predict(reduced_validation_set.dataset)

# - Evaluate model performance with MAE :
print(reduced_validation_set.evaluate_model_with_mean_absolute_error(prediction))
'''

# Impute columns with missing values method
# - Cut the dataset between a training set and a validation set :
imputed_training_set = Training_set(dataset)
imputed_validation_set = Validation_set(dataset)

# - Create the imputer :
imputer_1 = Imputer(imputer_type='simple', imputer_training_set=imputed_training_set.dataset)

# - impute : columns with missing val are dropped :
imputed_training_set.impute_columns_with_missing_val(imputer_1)
imputed_validation_set.impute_columns_with_missing_val(imputer_1)

'''
# - Use random forest regressor :
model = RandomForestRegressor(n_estimators=100, random_state=0)

# - Let's train the model :
model.fit(imputed_training_set.dataset, imputed_training_set.target)

# - Apply prediction :
prediction = model.predict(imputed_validation_set.dataset)

# - Evaluate model performance with MAE :
print(imputed_validation_set.evaluate_model_with_mean_absolute_error(prediction))
'''

# Impute and drop according to a specific number of missing values :
# - Cut the dataset between a training set and a validation set :
training_set = Training_set(dataset)
validation_set = Validation_set(dataset)

# Drop columns with missing values method - according to a limiter :
missing_val_number_limiter = 10
training_set.drop_columns_with_missing_val(missing_val_number_limiter)
validation_set.drop_columns_with_missing_val(missing_val_number_limiter)

# Locate missing values and put that information into columns :
training_set.locate_missing_values()
validation_set.locate_missing_values()

# - Create the imputer :
imputer_2 = Imputer(imputer_type='simple', imputer_training_set=training_set.dataset)

# Impute columns under limiter :
training_set.impute_columns_with_missing_val(imputer_2)
validation_set.impute_columns_with_missing_val(imputer_2)

# - Use random forest regressor :
model = RandomForestRegressor(n_estimators=100, random_state=0)

# - Let's train the model :
model.fit(training_set.dataset, training_set.target)

# - Apply prediction :
prediction = model.predict(validation_set.dataset)

# - Evaluate model performance with MAE :
print(validation_set.evaluate_model_with_mean_absolute_error(prediction))

