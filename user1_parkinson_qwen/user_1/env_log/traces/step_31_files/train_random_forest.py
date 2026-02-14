import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load the training data
train_data = pd.read_csv('train.csv')

# Load the data description for column information
with open('data_description.txt', 'r') as file:
    data_description = file.read()

# Split the data into features and target
features = train_data.drop(columns=['target'])
target = train_data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Save the predictions to a CSV file
predictions = pd.DataFrame({'predictions': y_pred})
predictions.to_csv('predictions_random_forest.csv', index=False)

# Save the trained model
joblib.dump(rf_regressor, 'random_forest_model.pkl')