import pandas as pd

# Load the submission file
submission = pd.read_csv('submission.csv')

# Assuming the test set is also available as a CSV file
test_set = pd.read_csv('test.csv')

# Extract the true labels from the test set
true_labels = test_set['label']

# Extract the predicted labels from the submission
predicted_labels = submission['prediction']

# Calculate the accuracy
accuracy = (predicted_labels == true_labels).mean()

print(f'Accuracy: {accuracy * 100:.2f}%')