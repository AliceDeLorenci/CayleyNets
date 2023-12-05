import torch
from torch_geometric.datasets import TUDataset
import networkx as nx
import matplotlib.pyplot as plt

from CayleyNet import CayleyConv
from importlib import reload  # Python 3.4+

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGPooling, TopKPooling
from torch_geometric.nn import global_mean_pool

from torch_geometric.datasets import Planetoid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes, r=64, sparse = False, dropout = 0.5, normalizartion = 'sym', bias = True):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = CayleyConv(num_node_features, hidden_channels, r, normalization = normalizartion, bias = bias, sparse = sparse)
        #self.pool = TopKPooling(hidden_channels, ratio=0.9)
        self.conv2 = CayleyConv(hidden_channels, hidden_channels, r, normalization = normalizartion, bias = bias, sparse = sparse)
        self.lin = Linear(hidden_channels, num_classes)
        self.sparse = sparse
        self.dropout = dropout
        self.normalizartion = normalizartion
        self.bias = bias


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        
        x = self.lin(x)
        
        return F.log_softmax(x.float(), dim=-1)
    


#Load cora dataset

dataset = Planetoid(root='data/Planetoid', name='Cora')
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
print(f'Contains self-loops: {data.contains_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')


# We will use the Cora dataset
data = dataset[0] 
data = data.to(device)


# Hyperparameters
batch_size = 1
num_epochs = 100
learning_rate = 0.01
hidden_channels = 16
poly_order = 5
num_node_features = dataset.num_features
num_classes = dataset.num_classes
sparse = False

model = GCN(hidden_channels, num_node_features, num_classes, poly_order,sparse = sparse, normalizartion= 'rw').to(device)
print(model)
print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
#parameters of the CayleyConv layer
print(f'Number of parameters: {sum(p.numel() for p in model.conv1.parameters())}')
print(f'Number of parameters: {sum(p.numel() for p in model.conv2.parameters())}')

train_dataset = data.x[data.train_mask]
test_dataset = data.x[data.test_mask]
val_dataset = data.x[data.val_mask]
train_dataset.data = train_dataset.data.to(device)
test_dataset.data = test_dataset.data.to(device)
val_dataset.data = val_dataset.data.to(device)

y_train = data.y[data.train_mask]
y_test = data.y[data.test_mask]
y_val = data.y[data.val_mask]
y_train = y_train.to(device)
y_test = y_test.to(device)
y_val = y_val.to(device)


# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# save loss values for plotting
loss_values = []
for e in range(num_epochs):
    # Compute output
    out = model(data.x, data.edge_index)
    # Compute loss
    loss = F.nll_loss(out[data.train_mask], y_train)
    loss_values.append(loss.item())
    # Perform backward pass
    loss.backward()
    # Perform optimization step
    optimizer.step()
    # Reset gradients
    optimizer.zero_grad()
    # Print loss
    print(f'Epoch: {e:03d}, Loss: {loss:.4f}')

# retreeve the convolutionals layers of the model (parameters c_i and c_o)
c1 = model.conv1.c
c2 = model.conv2.c

# Print the parameters
print(c1)
print(c2)

#plot the loss
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


model.eval()

data = dataset[0].to(device)
correct = 0

# Compute accuracy on the test set
pred = model(data.x, data.edge_index).argmax(dim=1)
correct += pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc = correct / data.test_mask.sum().item()
print(f'Test accuracy: {acc:.4f}')
