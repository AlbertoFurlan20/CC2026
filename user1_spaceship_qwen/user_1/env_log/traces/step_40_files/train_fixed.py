import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
train_df = pd.read_csv("train.csv")

# Print the first few rows to ensure the data is loaded correctly
print(train_df.head())

# Preprocess the data
# Assuming the last column is the target variable in the train set
target_column = 'Transported'

# Actual categorical columns should be specified based on the dataset
categorical_columns = ['Cabin', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP']
numeric_columns = [col for col in train_df.select_dtypes(include=[float, int]).columns if col != target_column]

# Define preprocessing for numeric columns (fill missing values and scale)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Define preprocessing for categorical columns (fill missing values and one-hot encode)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)])

# Split the data into training and testing sets
X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

# Create and train a logistic regression model pipeline
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', LogisticRegression(random_state=42))])

model_pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = model_pipeline.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print(f"Model Accuracy: {accuracy}")

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean()}")

# Save the trained model
dump(model_pipeline, 'logistic_regression_model.joblib')