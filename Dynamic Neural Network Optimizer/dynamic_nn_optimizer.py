import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset

class DynamicNN(nn.Module):
    def __init__(self, input_dim, layers, output_dim):
        """
        Initializes a neural network with dynamic architecture.
        Args:
            input_dim (int): Input feature size.
            layers (list): List of integers representing hidden layer sizes.
            output_dim (int): Output size (e.g., number of classes).
        """
        super(DynamicNN, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        for layer_size in layers:
            self.layers.append(nn.Linear(prev_dim, layer_size))
            self.layers.append(nn.ReLU())
            prev_dim = layer_size
        self.layers.append(nn.Linear(prev_dim, output_dim))
        self.layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_model(model, dataloader, criterion, optimizer, epochs=10):
    """
    Trains the neural network.
    """
    model.train()
    for epoch in range(epochs):
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def evaluate_model(model, dataloader):
    """
    Evaluates the neural network on test data.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

# Example Hyperparameter Grid Search:
# Create dummy data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=3)
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

param_grid = {
    "layers": [[64, 32], [128, 64, 32]],
    "lr": [0.01, 0.001],
    "epochs": [10, 20]
}

results = []
for params in ParameterGrid(param_grid):
    model = DynamicNN(input_dim=20, layers=params["layers"], output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    model = train_model(model, dataloader, criterion, optimizer, epochs=params["epochs"])
    accuracy = evaluate_model(model, dataloader)
    results.append({"params": params, "accuracy": accuracy})

best_model = max(results, key=lambda x: x["accuracy"])
print("Best Model:", best_model)
