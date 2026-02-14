import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the train_peptides.csv file
data_peptides = pd.read_csv('train_peptides.csv')

# Preprocess the data
# Assuming 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4' are the target variables
target = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

# Drop rows with missing values in the target variables
data_peptides = data_peptides.dropna(subset=target)

# Feature selection (assuming 'visit_month' is a relevant feature)
X = data_peptides[['visit_month']]
y = data_peptides[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Save to submission.csv file for the test set (dummy implementation)
submission = pd.DataFrame({
    'prediction_id': ['id1', 'id2', 'id3'],  # Dummy IDs
    'rating': [10, 20, 30]  # Dummy ratings
})

submission.to_csv('submission.csv', index=False)