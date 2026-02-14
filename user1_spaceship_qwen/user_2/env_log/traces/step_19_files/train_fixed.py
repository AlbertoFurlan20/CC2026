import seaborn as sns
import pandas as pd 
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np

def create_new_dataframe(data, column_names):
    new_data = {}
    
    for column in column_names:
        if column in data.columns:
            new_data[column] = data[column]
        else:
            new_data[column] = pd.Series(0, index=data.index)
    
    new_dataframe = pd.DataFrame(new_data)
    return new_dataframe

# Loading the dataset to train a binary classifier downstream
df = pd.read_csv("train.csv")
print(df.head())  # Print the first few rows of the dataframe to verify the data

# Handling missing values
imputer = SimpleImputer(strategy='most_frequent')
df[['Age', 'CryoSleep', 'VIP']] = imputer.fit_transform(df[['Age', 'CryoSleep', 'VIP']])

# Encoding categorical variables
label_encoders = {}
categorical_columns = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side']
for column in categorical_columns:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

# Expanding features to have boolean values as opposed to categorical
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
if 'Deck' in df.columns:
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df = df.drop('Cabin', axis=1)
else:
    df['Deck'] = pd.Series(0, index=df.index)  # Create a dummy column if 'Deck' does not exist

# Check if 'Side' column exists before applying pd.get_dummies
if 'Side' in df.columns:
    df = pd.get_dummies(df, columns=['Side'])
else:
    print(f"Error: Column 'Side' does not exist in the DataFrame. Stopping execution.")
    exit()

# Feature selection using SelectKBest with chi-squared test
X = df.drop('Transported', axis=1)
y = df['Transported']
selector = SelectKBest(chi2, k=10)
X_selected = selector.fit_transform(X, y)

# Splitting the data into features and labels
TargetY = df["Transported"]
ResourceX = X_selected

# Splitting the data into training and validation sets
train_data, val_data, TargetY_train, TargetY_val = train_test_split(ResourceX, TargetY, test_size=0.2, random_state=1)

# Define models to evaluate
models = [
    ('LogisticRegression', LogisticRegression()),
    ('RandomForest', RandomForestClassifier()),
    ('SVM', SVC(probability=True))
]

best_model = None
best_accuracy = 0

# Iterate over different models and select the best performing one based on validation accuracy
for name, model in models:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(train_data, TargetY_train)
    val_accuracy = pipeline.score(val_data, TargetY_val)
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_model = pipeline

print(f"Best Model: {best_model.named_steps['model'].__class__.__name__}")
print(f"Best Validation Accuracy: {best_accuracy}")

# Apply the best model to the test data
test_data = pd.read_csv('test.csv')
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop('Cabin', axis=1)

# Handling missing values in test data
test_data[['Age', 'CryoSleep', 'VIP']] = imputer.transform(test_data[['Age', 'CryoSleep', 'VIP']])

# Encoding categorical variables in test data
for column in categorical_columns:
    test_data[column] = label_encoders[column].transform(test_data[column].astype(str))

# Expanding features to have boolean values as opposed to categorical
test_X = pd.get_dummies(test_data[selectColumns], columns=['Side'])

# Feature selection on test data
test_X_selected = selector.transform(test_X)

# Make predictions
test_preds = best_model.predict(test_X_selected)

# Creating submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test