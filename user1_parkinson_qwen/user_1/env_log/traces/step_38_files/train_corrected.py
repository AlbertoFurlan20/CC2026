import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate_model(model, features, target):
    # Load the train_clinical_data.csv file
    data_clinical = pd.read_csv('train_clinical_data.csv')

    # Preprocess the data
    # Drop rows with missing values in the target variable
    data_clinical = data_clinical.dropna(subset=[target])

    # Feature selection
    X = data_clinical[features]
    y = data_clinical[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    n = len(predictions)
    s = 2 * np.abs(y_test - predictions).sum() / (np.abs(y_test) + np.abs(predictions)).sum()
    smape = s * 100 / n
    print(f'Mean Squared Error: {mse}')
    print(f'Symmetric Mean Absolute Percentage Error: {smape:.2f}%')

    # Save to submission.csv file for the test set (dummy implementation)
    submission = pd.DataFrame({
        'prediction_id': ['id1', 'id2', 'id3'],  # Dummy IDs
        'rating': [10, 20, 30]  # Dummy ratings
    })

    submission.to_csv('submission.csv', index=False)

# Example usage
if __name__ == "__main__":
    # Define model and features
    model = LinearRegression()
    features = ['visit_month']
    target = 'updrs_1'

    # Evaluate the model
    evaluate_model(model, features, target)