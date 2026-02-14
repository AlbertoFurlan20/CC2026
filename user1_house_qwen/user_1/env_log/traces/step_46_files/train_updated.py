import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the data, and separate the target
train_file_path = 'train.csv'
test_file_path = 'test.csv'
data_description_file_path = 'data_description.txt'

# Read data description to get feature descriptions
with open(data_description_file_path, 'r') as file:
    data_description = file.readlines()

# Define categorical and numerical columns based on data description
categorical_features = []
numerical_features = []

for line in data_description:
    if 'Categorical' in line:
        categorical_features.append(line.split(':')[0].strip())
    elif 'Numerical' in line:
        numerical_features.append(line.split(':')[0].strip())

# Load the training data
home_data = pd.read_csv(train_file_path)

# Separate the target variable
y = home_data.SalePrice

# Load the test data
test_data = pd.read_csv(test_file_path)

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define model
model = RandomForestRegressor(random_state=1)

# Create and fit the full pipeline
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('model', model)])

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(home_data[numerical_features + categorical_features], y, random_state=1)

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
test_X = test_data[numerical_features + categorical_features]
test_preds = full_pipeline.predict(test_X)

# Save predictions to file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)