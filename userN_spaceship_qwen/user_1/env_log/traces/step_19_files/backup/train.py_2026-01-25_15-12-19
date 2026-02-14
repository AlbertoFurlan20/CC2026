import seaborn as sns
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
# You can check all the features as column names and try to find good correlations with the target variable
selectColumns = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
ResourceX = pd.get_dummies(train_data[selectColumns])
ResourceX_test = pd.get_dummies(val_data[selectColumns])

# Splitting the data into features and target variable
X_train, X_val, y_train, y_val = train_test_split(ResourceX, TargetY, test_size=0.2, random_state=1)

# Initializing and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions on the validation set
val_predictions = model.predict(X_val)
train_predictions = model.predict(X_train)

# Calculating accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
val_accuracy = accuracy_score(y_val, val_predictions)

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