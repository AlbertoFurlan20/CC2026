import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, make_scorer
import numpy as np

# Load the data
peptides_df = pd.read_csv('train_peptides.csv')
proteins_df = pd.read_csv('train_proteins.csv')

# Merge the datasets based on a common column (assuming 'peptide_id' is the common column)
merged_df = pd.merge(peptides_df, proteins_df, on='peptide_id')

# Preprocess the data
X = merged_df.drop(columns=['protein_id', 'target'])
y = merged_df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using SMAPE
def smape(y_true, y_pred):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return np.mean(numerator / denominator) * 100.0

smape_score = smape(y_test, y_pred)
print(f'SMAPE: {smape_score:.2f}%')