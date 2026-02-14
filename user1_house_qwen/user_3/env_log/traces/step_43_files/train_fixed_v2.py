import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Preprocess features
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Fill missing values with the mean of each column
X = X.fillna(X.mean())

# Encode categorical features
categorical_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars']
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

# Combine preprocessing steps into a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', ordinal_encoder, categorical_features)])

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

# Fill missing values in test data with the mean of each column
test_X = test_X.fillna(test_X.mean())

# Encode categorical features in test data
test_X_encoded = ordinal_encoder.fit_transform(test_X)

# Make predictions on the test set
test_preds = model.predict(test_X_encoded)

# Create the submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)