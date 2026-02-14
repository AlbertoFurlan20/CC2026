# Import helpful libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

# Load the data, and separate the target
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

# You can change the features needed for this task depending on your understanding of the features and the final task
features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into testing and training data
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    valid_mae = mean_absolute_error(y_valid, model.predict(X_valid))
    return train_mae, valid_mae

# Define a list of models to try
models = [
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    LinearRegression(),
    DecisionTreeRegressor(),
    SVR()
]

# Define a list of feature sets to try
feature_sets = [
    features,
    ['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'],
    ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
]

# Initialize variables to store the best model and its MAE
best_model = None
best_mae = float('inf')

# Loop over the models and feature sets
for model in models:
    for feature_set in feature_sets:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)
        X_train = X_train[feature_set]
        X_valid = X_valid[feature_set]
        train_mae, valid_mae = train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid)
        
        # Print the MAE for this model and feature set
        print(f"Model: {type(model).__name__}, Feature Set: {feature_set}, Train MAE: {train_mae:.0f}, Validation MAE: {valid_mae:.0f}")
        
        # If this model and feature set have a lower MAE than the current best, update the best model and its MAE
        if valid_mae < best_mae:
            best_model = model
            best_mae = valid_mae

# Train the best model on the entire training set and make predictions on the test set
test_data = pd.read_csv('test.csv')
test_X = test_data[features]
best_model.fit(X, y)
test_preds = best_model.predict(test_X)

# Save the predictions to a submission file
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)