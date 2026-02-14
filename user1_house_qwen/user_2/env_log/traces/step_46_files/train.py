# Import helpful libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Preprocess the 'Shed' feature
# Assuming 'Shed' is a categorical feature that indicates whether a shed is present or not
home_data['Shed'] = home_data['Shed'].map({'Yes': 1, 'No': 0}).astype(int)

y = home_data.SalePrice

# You can change the features needed for this task depending on your understanding of the features and the final task
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'Shed']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Define the model
rf_model = RandomForestRegressor(random_state=1)
lr_model = LinearRegression()

# Fit the Random Forest model
rf_model.fit(train_X, train_y)

# Fit the Linear Regression model
lr_model.fit(train_X, train_y)

# Make predictions
train_rf_preds = rf_model.predict(train_X)
valid_rf_preds = rf_model.predict(valid_X)
train_lr_preds = lr_model.predict(train_X)
valid_lr_preds = lr_model.predict(valid_X)

# Calculate MAE for training data
train_rf_mae = mean_absolute_error(train_y, train_rf_preds)
train_lr_mae = mean_absolute_error(train_y, train_lr_preds)

# Calculate MAE for validation data
valid_rf_mae = mean_absolute_error(valid_y, valid_rf_preds)
valid_lr_mae = mean_absolute_error(valid_y, valid_lr_preds)

# Print the MAE metrics
print("Random Forest - Train MAE: {:,.0f}".format(train_rf_mae))
print("Random Forest - Validation MAE: {:,.0f}".format(valid_rf_mae))
print("Linear Regression - Train MAE: {:,.0f}".format(train_lr_mae))
print("Linear Regression - Validation MAE: {:,.0f}".format(valid_lr_mae))

# Load the test data
test_data = pd.read_csv('test.csv')

# Preprocess the 'Shed' feature in the test data
test_data['Shed'] = test_data['Shed'].map({'Yes': 1, 'No': 0}).astype(int)

test_X = test_data[features]
test_rf_preds = rf_model.predict(test_X)
test_lr_preds = lr_model.predict(test_X)

# Create the submission file
output_rf = pd.DataFrame({'Id': test_data.Id,
                          'SalePrice': test_rf_preds})
output_lr = pd.DataFrame({'Id': test_data.Id,
                          'SalePrice': test_lr_preds})
output_rf.to_csv('submission_rf.csv', index=False)
output_lr.to_csv('submission_lr.csv', index=False)