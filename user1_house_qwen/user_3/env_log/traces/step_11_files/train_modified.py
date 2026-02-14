# Import helpful libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

# Check if the MSSubClass feature is already converted to categorical. If not, convert it using pandas' `astype('category')` method.
if home_data['MSSubClass'].dtype != 'category':
    home_data['MSSubClass'] = home_data['MSSubClass'].astype('category')

y = home_data.SalePrice

# You can change the features needed for this task depending on your understanding of the features and the final task
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(train_X, train_y)

# Make predictions on the validation set
valid_y_pred = model.predict(valid_X)

# Calculate the Mean Absolute Error (MAE) for both training and validation sets
train_mae = mean_absolute_error(train_y, model.predict(train_X))
valid_mae = mean_absolute_error(valid_y, valid_y_pred)

# Print the MAE for both training and validation sets
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Load the test data
test_data = pd.read_csv('test.csv')
test_X = test_data[features]

# Convert the MSSubClass feature in the test data to categorical if it's not already
if test_data['MSSubClass'].dtype != 'category':
    test_data['MSSubClass'] = test_data['MSSubClass'].astype('category')

# Make predictions on the test set
test_preds = model.predict(test_X)

# Create the submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)