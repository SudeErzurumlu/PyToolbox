import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, DynamicEdgeConv
from torch_geometric.data import Data

class DynamicGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes a dynamic graph neural network with GCN layers.
        """
        super(DynamicGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        """
        Forward pass for the GNN.
        """
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Example Usage
node_features = torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float)
edge_indices = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Undirected graph
graph_data = Data(x=node_features, edge_index=edge_indices)

model = DynamicGNN(input_dim=2, hidden_dim=4, output_dim=2)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(graph_data.x, graph_data.edge_index)
    loss = criterion(output, graph_data.x)  # Self-supervised example
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
