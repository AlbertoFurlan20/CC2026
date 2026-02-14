import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load the trained model
model = torch.load('trained_model.pth')
model.eval()

# Initialize dataframe for submission
submission = pd.DataFrame()

# Make predictions
for data, _ in test_loader:
    outputs = model(data)
    probabilities = torch.softmax(outputs, dim=1)
    class_probabilities = {i: prob.item() for i, prob in enumerate(probabilities[0])}
    submission = submission.append(class_probabilities, ignore_index=True)

# Save submission to csv
submission.to_csv('submission.csv', index=False)