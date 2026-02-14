import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Define categorical and numerical columns
categorical_features = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour']
numerical_features = ['LotArea', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Function to load data description
def load_data_description(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    features = {}
    for line in lines:
        if line.strip() and not line.startswith('#'):
            feature, description = line.split(': ', 1)
            features[feature.strip()] = description.strip()
    return features

# Load data description
data_description = load_data_description('data_description.txt')

# Define a function to select features based on descriptions
def select_features(data_description, features):
    selected_features = []
    for feature in features:
        if data_description.get(feature) and 'important' in data_description[feature].lower():
            selected_features.append(feature)
    return selected_features

# Update numerical and categorical features based on feature selection
selected_numerical_features = select_features(data_description, numerical_features)
selected_categorical_features = select_features(data_description, categorical_features)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_regression, k=len(selected_numerical_features)))])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, selected_numerical_features),
        ('cat', categorical_transformer, selected_categorical_features)])

# Define model
model = RandomForestRegressor(random_state=1)

# Create and fit the full pipeline
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(home_data[selected_numerical_features + selected_categorical_features], y, random_state=1)

# Fit the model
full_pipeline.fit(train_X, train_y)

# Make predictions
train_pred = full_pipeline.predict(train_X)
valid_pred = full_pipeline.predict(valid_X)

train_mae = mean_absolute_error(train_y, train_pred)
valid_mae = mean_absolute_error(valid_y, valid_pred)

# Print MAE
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Prepare test data
test_data = pd.read_csv('test.csv')
test_X = test_data[selected_numerical_features + selected_categorical_features]
test_preds = full_pipeline.predict(test_X)

# Save predictions to file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)