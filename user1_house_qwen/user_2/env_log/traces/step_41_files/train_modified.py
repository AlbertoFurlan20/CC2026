import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Identify categorical and numerical features
categorical_features = ['Shed']
numerical_features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features, and preview the data
X = home_data[numerical_features + categorical_features]

# Ensure 'Shed' is correctly processed as a categorical variable
X['Shed'] = X['Shed'].astype('category')

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Define the preprocessing for categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor(random_state=1))])

# Fit the model
clf.fit(train_X, train_y)

# Make predictions
train_preds = clf.predict(train_X)
valid_preds = clf.predict(valid_X)

# Calculate MAE for training data
train_mae = mean_absolute_error(train_y, train_preds)

# Calculate MAE for validation data
valid_mae = mean_absolute_error(valid_y, valid_preds)

# Print the MAE metrics
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Load the test data
test_data = pd.read_csv('test.csv')
test_X = test_data[numerical_features + categorical_features]
test_X['Shed'] = test_X['Shed'].astype('category')  # Ensure 'Shed' is correctly processed as a categorical variable
test_preds = clf.predict(test_X)

# Create the submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)