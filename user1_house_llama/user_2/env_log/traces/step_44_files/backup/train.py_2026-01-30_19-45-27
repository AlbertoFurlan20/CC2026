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

# Function to load the data, separate the target, select the features, split the data into training and validation sets
def load_data(iowa_file_path):
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    X = home_data[features]
    train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)
    return train_X, valid_X, train_y, valid_y

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid):
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    valid_mae = mean_absolute_error(y_valid, model.predict(X_valid))
    return train_mae, valid_mae

# Function to train the best model on the entire training set and make predictions on the test set
def train_best_model(best_model, X, y, test_data):
    test_X = test_data[['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']]
    best_model.fit(X, y)
    test_preds = best_model.predict(test_X)
    return test_preds

# Function to save the predictions to a submission file
def save_predictions(test_preds, test_data):
    output = pd.DataFrame({'Id': test_data.Id,
                           'SalePrice': test_preds})
    output.to_csv('submission.csv', index=False)

# Load the data, separate the target, select the features, split the data into training and validation sets
iowa_file_path = 'train.csv'
train_X, valid_X, train_y, valid_y = load_data(iowa_file_path)

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
    ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'],
    ['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'],
    ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
]

# Initialize variables to store the best model and its MAE
best_model = None
best_mae = float('inf')

# Loop over the models and