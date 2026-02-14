import seaborn as sns
import pandas as pd 
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

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
    ResourceX = pd.get_dummies(df[selectColumns])
else:
    print(f"Error: Column 'Side' does not exist in the DataFrame. Stopping execution.")
    exit()

# Splitting the data into features and labels
TargetY = df["Transported"]
ResourceX = pd.get_dummies(df[selectColumns])

# Splitting the data into training and validation sets
train_data, val_data, TargetY_train, TargetY_val = train_test_split(ResourceX, TargetY, test_size=0.2, random_state=1)

# Training the model
model = LogisticRegression()
model.fit(train_data, TargetY_train)
train_accuracy = model.score(train_data, TargetY_train)
val_accuracy = model.score(val_data, TargetY_val)

print(f"Train Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")

test_data = pd.read_csv('test.csv')
test_data[["Deck", "Cabin_num", "Side"]] = test_data["Cabin"].str.split("/", expand=True)
test_data = test_data.drop('Cabin', axis=1)

# Handling missing values in test data
test_data[['Age', 'CryoSleep', 'VIP']] = imputer.transform(test_data[['Age', 'CryoSleep', 'VIP']])

# Encoding categorical variables in test data
for column in categorical_columns:
    test_data[column] = label_encoders[column].transform(test_data[column].astype(str))

# Expanding features to have boolean values as opposed to categorical
test_X = pd.get_dummies(test_data[selectColumns])
test_X.insert(loc = 17,
              column = 'Deck_T',
              value = 0)

# Making predictions
test_preds = model.predict(test_X)

# Creating submission file
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Transported': test_preds})
output.to_csv('submission.csv', index=False)