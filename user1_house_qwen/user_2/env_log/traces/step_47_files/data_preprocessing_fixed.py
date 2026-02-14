import pandas as pd

# Sample data
data = {
    'Shed': ['yes', 'no', 'yes', 'no'],
    'Age': [25, 30, 35, 40],
    'Income': [50000, 60000, 70000, 80000]
}

df = pd.DataFrame(data)

# Convert 'Shed' to categorical variable
df['Shed'] = df['Shed'].astype('category')

# Check if all expected features are present in the DataFrame
expected_features = ['Shed', 'Age', 'Income']
if all(feature in df.columns for feature in expected_features):
    print("All expected features are present.")
else:
    print("Missing features in the DataFrame.")

# Continue with further operations...