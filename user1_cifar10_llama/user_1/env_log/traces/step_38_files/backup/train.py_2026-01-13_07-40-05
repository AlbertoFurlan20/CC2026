# Import necessary libraries
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd

# Define the model class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the convolutional and pooling layers
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Define the fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Apply the convolutional and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output
        x = torch.flatten(x, 1)
        # Apply the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the data augmentation transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Define the dataloaders
batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)

# Define the optimizer and loss function
model = Net().to(device)
# Use Adam optimizer with a learning rate of 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Use cross-entropy loss function
criterion = nn.CrossEntropyLoss()

# Define a function to test the model
def test_model(dataloader):
    # Set the model to evaluation mode
    model.eval()
    # Initialize variables to keep track of the correct predictions
    correct = 0
    total = 0
    with torch.no_grad():
        # Iterate over the dataloader
        for inputs, labels in dataloader:
            # Move the inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Get the outputs from the model
            outputs = model(inputs)
            # Get the predicted labels
            _, predicted = torch.max(outputs.data, 1)
            # Update the total and correct variables
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return the accuracy
    return 100 * correct / total

# Set the device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train the model
epochs = 10
for epoch in range(epochs):
    # Initialize the running loss
    running_loss = 0.0
    # Set the model to training mode
    model.train()
    # Iterate over the train dataloader
    for i, (inputs, labels) in enumerate(train_dataloader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        # Get the outputs from the model
        outputs = model(inputs)

        # Calculate the loss
        loss = criterion(outputs, labels)
        # Backpropagate the loss
        loss.backward()
        # Update the model parameters
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()
        # Print the loss every 100 mini-batches
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # Test the model on the train and test datasets
    train_accuracy = test_model(train_dataloader)
    test_accuracy = test_model(test_dataloader)
    print(f'Epoch [{epoch+1}/{epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Print the final training and test accuracy
train_accuracy = test_model(train_dataloader)
test_accuracy = test_model(test_dataloader)
print(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

# Save the predictions to submission.csv
submission = pd.DataFrame(columns=list(range(10)), index=range(len(test_dataset)))
model.eval()
# Iterate over the test dataset
for idx, data in enumerate(test_dataset):
    # Move the input to the device
    inputs = data[0].unsqueeze(0).to(device)
    # Get the output from the model
    pred = model(inputs)
    # Get the predicted probabilities
    pred = torch.softmax(pred[0], dim=0)
    # Save the predicted probabilities to the submission dataframe
    submission.loc[idx] = pred.tolist()
# Save the submission dataframe to a csv file
submission.to_csv('submission.csv')