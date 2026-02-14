import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# Add all relevant features mentioned in data_description.txt to the features list
with open('data_description.txt', 'r') as file:
    features_description = file.readlines()

features = []
for line in features_description:
    if 'Feature' in line:
        feature_name = line.split(':')[0].strip()
        features.append(feature_name)

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Define the model
model = GradientBoostingRegressor(random_state=1)

# Fit the model
model.fit(train_X, train_y)

# Make predictions
train_preds = model.predict(train_X)
valid_preds = model.predict(valid_X)

# Calculate MAE for training data
train_mae = mean_absolute_error(train_y, train_preds)

# Calculate MAE for validation data
valid_mae = mean_absolute_error(valid_y, valid_preds)

# Print the MAE metrics
print("Train MAE: {:,.0f}".format(train_mae))
print("Validation MAE: {:,.0f}".format(valid_mae))

# Load the test data
test_data = pd.read_csv('test.csv')
test_X = test_data[features]
test_preds = model.predict(test_X)

# Create the submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)