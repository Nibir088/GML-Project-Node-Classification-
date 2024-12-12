import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class Graphormer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_pred):
        super(Graphormer, self).__init__()
        self.use_pred = use_pred
        if self.use_pred:
            self.encoder = torch.nn.Embedding(out_channels+1, hidden_channels)
        
        self.convs = torch.nn.ModuleList()
        self.attns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # Define the Graphormer layers
        self.convs.append(TransformerConv(in_channels, hidden_channels, heads=1, dropout=dropout))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_channels, hidden_channels, heads=1, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        self.convs.append(TransformerConv(hidden_channels, out_channels, heads=1, dropout=dropout))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        if self.use_pred:
            x = self.encoder(x)
            x = torch.flatten(x, start_dim=1)

        # Apply Graphormer layers (TransformerConv)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        # You can apply additional transformations if needed
        return x
