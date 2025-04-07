import sys
import os

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

def extract_feature_values_from_file(training_data_file_name, testing_data_file_name):
  train_df = pd.read_csv(os.path.join(os.getcwd(), training_data_file_name))
  test_df = pd.read_csv(os.path.join(os.getcwd(), testing_data_file_name))

  [training_features_set, training_target_set] = convert_to_features_and_target(train_df, is_training_data=True)
  [testing_features_set, _] = convert_to_features_and_target(test_df, is_training_data=False)

  return [training_features_set, training_target_set, testing_features_set]

def train_model(training_features_set, training_target_set):
  print(f'Training model using optimal tuning parameters...\n')

  # These are the optimal paramaters from tuning the Random Forest Classifier using RamdomizedSearchCV in the jupyter notebook
  optimal_tuning_parameter_values = {
    'n_estimators': 500, 
    'min_samples_split': 5, 
    'min_samples_leaf': 1, 
    'max_features': 'sqrt', 
    'max_depth': 110, 
    'criterion': 'entropy', 
    'bootstrap': False
  }

  model = RandomForestClassifier(**optimal_tuning_parameter_values)
  model.fit(training_features_set, training_target_set)
  
  print(f'Model training complete.\n')
  return model

def get_model_predictions_for_test_data(model, testing_features_set):
  model_predictions = model.predict(testing_features_set)
  print(f'Generated model predictions for test data.\n')
  return model_predictions

def write_model_predictions_to_file(model_predictions, file_name='model_predictions.txt'):
  print(f'Writing model predictions to file {file_name}...\n')
  with open(file_name, 'w') as f:
    for prediction in model_predictions:
      f.write(f"{prediction}\n")
  
  print(f'Model predictions written to file: {file_name}\n')

def convert_to_features_and_target(df: DataFrame, is_training_data=True):
  # Create a dataframe for the set of descriptive features
  features_df = df.iloc[:, 1:-1] if is_training_data else df.iloc[:, 1:]
  features_set = features_df.values  
  # Create a dataframe for the set of target features - only for testing dataset where the target feature column exists
  target_df = df.iloc[:, -1] if is_training_data else None
  target_set = target_df.values if target_df is not None else None
  
  return [features_set, target_set]

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print("Usage: script.py <training data file name>.csv <testing data file name>.csv.\nNote: the file must be in the same directory as this script.")
    sys.exit(1)

  training_data_file_name = sys.argv[1]
  testing_data_file_name = sys.argv[2]

  [training_features_set, training_target_set, testing_features_set] = extract_feature_values_from_file(training_data_file_name, testing_data_file_name)
  model = train_model(training_features_set, training_target_set)
  model_predictions = get_model_predictions_for_test_data(model, testing_features_set)
  write_model_predictions_to_file(model_predictions)
