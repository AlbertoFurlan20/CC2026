import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.nn import functional as F

# Define the data loader for the test set
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load the trained model
model = torch.load('model.pth')

# Set the model to evaluation mode
model.eval()

# Initialize lists to store true labels and predicted probabilities
true_labels = []
predicted_probabilities = []

# Make predictions on the test set
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        predicted_probabilities.extend(probabilities.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Save the per-class probabilities to submission.csv
import pandas as pd
submission = pd.DataFrame(predicted_probabilities, columns=[f'class_{i}' for i in range(10)])
submission['true_label'] = true_labels
submission.to_csv('submission.csv', index=False)