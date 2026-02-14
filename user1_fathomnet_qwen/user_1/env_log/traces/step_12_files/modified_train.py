import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from sklearn.metrics import hamming_loss, f1_score
from torchmetrics import Functional

# Define the neural network model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_layers(x)
        return logits

# Load the dataset
def load_data():
    transform = Compose([
        Resize((28, 28)),
        ToTensor(),
        Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = ImageFolder(root='data/train', transform=transform)
    test_dataset = ImageFolder(root='data/test', transform=transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Evaluation function
def test_loop(dataloader, model, loss_fn, metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            y_true.append(y.numpy())
            y_pred.append(pred.sigmoid().round().numpy())

    test_loss /= num_batches
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    hamming_loss_value = hamming_loss(y_true, y_pred)
    f1_score_value = f1_score(y_true, y_pred, average='macro')

    print(f"Test Error: \n Hamming Loss: {hamming_loss_value:.4f}, F1 Score: {f1_score_value:.4f} \n")

# Main function
def main():
    model = SimpleCNN()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    train_loader, test_loader = load_data()

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(test_loader, model, loss_fn, metric=Functional.hamming_loss)
    print("Done!")

if __name__ == "__main__":
    main()