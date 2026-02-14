import seaborn as sns
import pandas as pd 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

def create_new_dataframe(data, column_names):
    new_data = {}
    
    for column in column_names:
        if column in data.columns:
            new_data[column] = data[column]
        else:
            new_data[column] = pd.Series(0, index=data.index)
    
    new_dataframe = pd.DataFrame(new_data)
    return new_dataframe

# Loading the dataset to train a binary classfier downstream
df = pd.read_csv("train.csv")
num_examples = df.shape[0]
df = df.sample(frac = 1, random_state=1)
train_data = df[0:int(0.8*num_examples)]
val_data = df[int(0.8*num_examples)+1:]

train_data[["Deck", "Cabin_num", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data = train_data.drop('Cabin', axis=1) 

val_data[["Deck", "Cabin_num", "Side"]] = val_data["Cabin"].str.split("/", expand=True)
val_data = val_data.drop('Cabin', axis=1)

TargetY = train_data["Transported"]
TargetY_test = val_data["Transported"]

# Expanding features to have boolean values as opposed to categorical
# You can check all the features as column names and try to find good correlations with the target variable
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
ResourceX = pd.get_dummies(train_data[selectColumns])
ResourceX_test = pd.get_dummies(val_data[selectColumns])

# Define a logistic regression model and train it on the dataset
model = LogisticRegression()
model.fit(ResourceX, TargetY)

# Make predictions on the training and validation sets
train_preds = model.predict(ResourceX)
val_preds = model.predict(ResourceX_test)

# Calculate the accuracy of the model on the training and validation sets
train_accuracy = accuracy_score(TargetY, train_preds)
val_accuracy = accuracy_score(TargetY_test, val_preds)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

# Feature selection using SelectFromModel
selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
selector.fit(ResourceX, TargetY)
support = selector.get_support()
feature_names = ResourceX.columns[support]
ResourceX_selected = ResourceX[feature_names]
ResourceX_test_selected = ResourceX_test[feature_names]

model_selected = LogisticRegression()
model_selected.fit(ResourceX_selected, TargetY)
train_preds_selected = model_selected.predict(ResourceX_selected)
val_preds_selected = model_selected.predict(ResourceX_test_selected)

train_accuracy_selected = accuracy_score(TargetY, train_preds_selected)
val_accuracy_selected = accuracy_score(TargetY_test, val_preds_selected)

print(f"Train Accuracy after feature selection: {train_accuracy_selected}")
print(f"Validation Accuracy after feature selection: {val_accuracy_selected}")

# Grid search for hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'max_iter': [500, 1000, 1500]
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(ResourceX, TargetY)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best accuracy: {grid_search.best_score_}")

# Train a random forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(ResourceX, TargetY)
rf_train_preds = rf_model.predict(ResourceX)
rf_val_preds = rf_model.predict(ResourceX_test)

rf_train_accuracy = accuracy_score(TargetY, rf_train_preds)
rf_val_accuracy = accuracy_score(TargetY_test, rf_val_preds)

print(f"Train Accuracy for Random Forest: {rf_train_accuracy}")
print(f"Validation Accuracy for Random Forest: {rf_val_accuracy}")

# Train a gradient boosting model
gb_model = GradientBoostingClassifier(n_estimators=100)
gb_model.fit(ResourceX, TargetY)
gb_train_preds = gb_model.predict(ResourceX)
gb_val_preds = gb_model.predict(ResourceX_test)

gb_train_accuracy = accuracy_score(TargetY, gb_train_preds)
gb_val_accuracy = accuracy_score(TargetY_test, gb_val_preds)

print(f"Train Accuracy for Gradient Boosting: {gb_train_accuracy}")
print(f"Validation Accuracy for Gradient Boosting: {gb_val_accuracy}")

# Train a support vector machine model
svm_model = SVC()
svm_model.fit(ResourceX,