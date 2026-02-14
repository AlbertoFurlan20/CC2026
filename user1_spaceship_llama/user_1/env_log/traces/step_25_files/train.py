import seaborn as sns
import pandas as pd 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline

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

# Define a pipeline with a scaler and a classifier
scaler = StandardScaler()
selector = SelectKBest(mutual_info_classif, k=10)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', selector),
    ('classifier', classifier)
])

# Define hyperparameters to tune
param_grid = {
    'selector__k': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(ResourceX, TargetY)

# Get the best model and its accuracy
best_model = grid_search.best_estimator_
train_accuracy = grid_search.best_score_
val_accuracy = accuracy_score(TargetY_test, best_model.predict(ResourceX_test))

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

test_data = pd.read_csv('test.csv')
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop('Cabin', axis=1)

test_X = pd.get_dummies(test_data[selectColumns])
test_X.insert(loc = 17,
          column = 'Deck_T',
          value = 0)

test_preds = best_model.predict(test_X)


output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds})
output.to_csv('submission.csv', index=False)

# Iterate over different models or feature selections to get a better performance
# Try different feature selection methods
selector2 = SelectKBest(mutual_info_classif, k=5)
selector3 = SelectKBest(mutual_info_classif, k=15)
selector4 = SelectFdr(fdr=0.05)

pipeline2 = Pipeline([
    ('scaler', scaler),
    ('selector2', selector2),
    ('classifier', classifier)
])

pipeline3 = Pipeline([
    ('scaler', scaler),
    ('selector3', selector3),
    ('classifier', classifier)
])

pipeline4 = Pipeline([
    ('scaler', scaler),
    ('selector4', selector4),
    ('classifier', classifier)
])

param_grid2 = {
    'selector2__k': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200]
}

param_grid3 = {
    'selector3__k': [5, 10, 15],
    'classifier__n_estimators': [50, 100, 200]
}

param_grid4 = {
    'selector4__fdr': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [50, 100, 200]
}

grid_search2 = GridSearchCV(pipeline2, param_grid2, cv=5, scoring='accuracy')
grid_search3 = GridSearchCV(pipeline3, param_grid3, cv=5, scoring='accuracy')
grid_search4 = GridSearchCV(pipeline4, param_grid4, cv=5, scoring='accuracy')

grid_search2.fit(ResourceX, TargetY)
grid_search3.fit(ResourceX, TargetY)
grid_search4.fit(ResourceX, TargetY)

best_model2 = grid_search2.best_estimator_
best_model3 = grid_search3.best_estimator_
best_model4 = grid_search4.best_estimator_

train_accuracy2 = grid_search2.best_score_
train_accuracy3 = grid_search3.best_score_
train_accuracy4 = grid_search4.best_score_

val_accuracy2 = accuracy_score(TargetY_test, best_model2.predict(ResourceX_test))
val_accuracy3 = accuracy_score(TargetY_test, best_model3.predict(ResourceX_test))
val_accuracy4 = accuracy_score(TargetY_test, best_model4.predict(ResourceX_test))

print(f"Train Accuracy (selector2): {train_accuracy2}")
print(f"Validation Accuracy (selector2): {val_accuracy2}")

print(f"Train Accuracy (selector3): {train_accuracy3}")
print(f"Validation Accuracy (selector3): {val_accuracy3}")

print(f"Train Accuracy (selector4): {train_accuracy4}")
print(f"Validation Accuracy (selector4): {val_accuracy4}")

test_X2 = pd.get_dummies(test_data[selectColumns])
test_X2.insert(loc = 17,
          column = 'Deck_T',
          value = 0)

test_X3 = pd.get_dummies(test_data[selectColumns])
test_X3.insert(loc = 17,
          column = 'Deck_T',
          value = 0)

test_X4 = pd.get_dummies(test_data[selectColumns])
test_X4.insert(loc = 17,
          column = 'Deck_T',
          value = 0)

test_preds2 = best_model2.predict(test_X2)
test_preds3 = best_model3.predict(test_X3)
test_preds4 = best_model4.predict(test_X4)

output2 = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds2})
output2.to_csv('submission2.csv', index=False)

output3 = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds3})
output3.to_csv('submission3.csv', index=False)

output4 = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds4})
output4.to_csv('submission4.csv', index=False)