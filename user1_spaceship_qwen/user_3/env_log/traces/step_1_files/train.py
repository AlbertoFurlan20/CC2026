import seaborn as sns
import pandas as pd 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

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
num_examples = df.shape[0]
df = df.sample(frac=1, random_state=1)
train_data = df[0:int(0.8 * num_examples)]
val_data = df[int(0.8 * num_examples) + 1:]

train_data[["Deck", "Cabin_num", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data = train_data.drop('Cabin', axis=1) 

val_data[["Deck", "Cabin_num", "Side"]] = val_data["Cabin"].str.split("/", expand=True)
val_data = val_data.drop('Cabin', axis=1)

TargetY = train_data["Transported"]
TargetY_test = val_data["Transported"]

# Expanding features to have boolean values as opposed to categorical
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
ResourceX = pd.get_dummies(train_data[selectColumns])
ResourceX_test = pd.get_dummies(val_data[selectColumns])

# Splitting the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(ResourceX, TargetY, test_size=0.2, random_state=42)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating the model
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)

# Saving the trained model
joblib.dump(model, 'logistic_regression_model.pkl')

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

test_data = pd.read_csv('test.csv')
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop('Cabin', axis=1)

test_X = pd.get_dummies(test_data[selectColumns])
test_X.insert(loc=17, column='Deck_T', value=0)

test_preds = model.predict(test_X)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds})
output.to_csv('submission.csv', index=False)