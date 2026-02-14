import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torchvision.transforms as transforms
import torchvision.models as models

# Define the model architecture using a pre-trained VGG16
class PretrainedVGG16(nn.Module):
    def __init__(self, num_classes):
        super(PretrainedVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        # Replace the last fully connected layer
        self.vgg16.classifier[6] = nn.Linear(self.vgg16.classifier[6].in_features, num_classes)

    def forward(self, x):
        x = self.vgg16(x)
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
            loss = criterion(outputs, labels.long())  # Ensure labels are long type
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

# Data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create data loaders
train_dataset = ImageDataset(train_df, transform=data_transforms['train'])
valid_dataset = ImageDataset(valid_df, transform=data_transforms['valid'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model, criterion, optimizer
model = PretrainedVGG16(num_classes)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer)

# Save the trained model
torch.save(model.state_dict(), 'pretrained_vgg16_model.pth')