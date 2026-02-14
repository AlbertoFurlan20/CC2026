import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Preprocess features
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Handle missing values
imputer = SimpleImputer(strategy='median')

# Encode categorical features
categorical_features = ['MSSubClass', 'OverallCond', 'MoSold', 'YrSold']
ordinal_encoder = OrdinalEncoder()

# Combine preprocessing steps into a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), [i for i in range(X.shape[1]) if X.dtypes[i] != 'O']),
        ('cat', OrdinalEncoder(), categorical_features)])

# Apply the preprocessor to the data
X_preprocessed = preprocessor.fit_transform(X)

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X_preprocessed, y, random_state=1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(train_X, train_y)

# Make predictions on the validation set
valid_y_pred = model.predict(valid_X)

# Calculate the Mean Squared Error (MSE) for the validation set
valid_mse = mean_squared_error(valid_y, valid_y_pred)

# Print the MSE for the validation set
print("Validation MSE: {:,.0f}".format(valid_mse))

# Load the test data
test_data = pd.read_csv('test.csv')
test_X = test_data[features]

# Handle missing values in test data
test_X_imputed = imputer.transform(test_X)

# Encode categorical features in test data
test_X_encoded = ordinal_encoder.transform(test_X_imputed[:, [1, 3, 24]])

# Make predictions on the test set
test_preds = model.predict(test_X_encoded)

# Create the submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)