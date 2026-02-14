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

# Function to evaluate the performance of a model and feature set
def evaluate_model(model, X_train, y_train, X_valid, y_valid):
    train_mae, valid_mae = train_and_evaluate_model(model, X_train, y_train, X_valid, y_valid)
    print(f'Train MAE: {train_mae}, Valid MAE: {valid_mae}')
    return train_mae, valid_mae

# Function to try different models and feature sets
def try_models_and_feature_sets(train_X, valid_X, train_y, valid_y):
    models = [
        RandomForestRegressor(),
        GradientBoostingRegressor(),
        LinearRegression(),
        DecisionTreeRegressor(),
        SVR()
    ]

    feature_sets = [
        ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'],
        ['MSSubClass', 'LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'],
        ['LotArea', 'OverallQual', 'YearBuilt', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    ]

    for model in models:
        for feature_set in feature_sets:
            X_train_feature_set = train_X[feature_set]
            X_valid_feature_set = valid_X[feature_set]
            train_mae, valid_mae = evaluate_model(model, X_train_feature_set, train_y, X_valid_feature_set, valid_y)
            print(f'Model: {type(model).__name__}, Feature Set: {feature_set}, Train MAE: {train_mae}, Valid MAE: {valid_mae}')

# Function to train the best model on the entire training set and make predictions on the test set
def train_best_model(best_model, X, y, test_data):
    test_X = test_data[['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGr