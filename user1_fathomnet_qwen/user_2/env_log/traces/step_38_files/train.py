import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# Define the model architecture
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preprocessing and loading
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.iloc[idx]['id']
        image_path = f"images/{image_id}.jpg"
        image = load_image(image_path)  # Assume this function loads the image
        label = self.df.iloc[idx]['label']  # Assuming 'label' is the target column
        if self.transform:
            image = self.transform(image)
        return image, label

# Training loop
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        running_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            print(f"Validation Accuracy: {accuracy:.2f}%")

# Load data
train_df = pd.read_csv("train.csv")
valid_df = pd.read_csv("valid.csv")

# Define parameters
batch_size = 32
num_workers = 4
learning_rate = 0.001
num_classes = train_df['label'].nunique()  # Determine the number of classes

# Create data loaders
train_dataset = ImageDataset(train_df)
valid_dataset = ImageDataset(valid_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model, criterion, optimizer
model = SimpleCNN(num_classes)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), 'simple_cnn_model.pth')

# Predict on validation set
proba = []
with torch.no_grad():
    model.eval()
    for images, _ in valid_loader:
        outputs = model(images)
        probabilities = torch.sigmoid(outputs)  # Use sigmoid for multi-label classification
        proba.extend(probabilities.cpu().numpy())

predict = []
for p in proba:
    predict.append([str(int(i > 0.5)) for i in p])  # Convert probabilities to binary