import os.path as osp

import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, convert)
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

#setting manuel seed
torch.manual_seed(12345)

#loading the edge list
g = nx.read_edgelist('C:/Users/DELL/Desktop/yeast.edgelist')

#converting the data into a torch readable mode
g_torch = convert.from_networkx(g)

data = g_torch

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.02, test_ratio=0.02 )

#creating the adjacency matrix of the edgelist
adj = nx.adjacency_matrix(g)

#preprocessing the graph
adj = sp.coo_matrix(adj)
adj_ = adj + sp.eye(adj.shape[0])
rowsum = np.array(adj_.sum(1))
degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

#normalising the adjacency matrix
adj_norm_np = adj_normalized.toarray()

#converting the normalised adjacency matrix into a torch tensor
data.x = torch.tensor(adj_norm_np).float()


#defining the Graph Convolution Network layers
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(6526, 32)
        self.conv2 = GCNConv(32, 16)

    def forward(self, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
        x = self.conv2(x, data.train_pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)


#giving the input to the GCN layers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
model = model.float()

#GCN Optimiser
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

#defining the link lables 
def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


#defining the training process
def train():
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes = x.size(0))

    neg_edge_index = negative_sampling(
        edge_index = pos_edge_index_with_self_loops, num_nodes = x.size(0),
        num_neg_samples = pos_edge_index.size(1))

    link_logits = model(pos_edge_index, neg_edge_index) #prediction
    link_labels = get_link_labels(pos_edge_index, neg_edge_index) 

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


#defining the testing process
def test():
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]
        link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index))
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()
        perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs


#Training the GCN Model and Evaluating its efficiency
best_val_perf = test_perf = 0
for epoch in range(1, 10):
    train_loss = train()
    val_perf, tmp_test_perf = test()
    if val_perf > best_val_perf:
        best_val_perf = val_perf
        test_perf = tmp_test_perf
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_loss, best_val_perf, test_perf))