import os
import sys
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, JumpingKnowledge, Linear
from layers import GraphAttentionLayer, InnerProductDecoder


class GSATSRBP(nn.Module):
    """
    A Graph Neural Network model combining GraphSAGE, GAT, with a decoding layer for reconstruction.
    This model aims for drug-circRNA interaction prediction.
    """

    def __init__(self, n_in_features: int, n_hid_layers: int, hid_features: list, n_heads: list,
                 n_drug: int, n_cir: int, add_layer_attn: bool, residual: bool, dropout: float = 0.6):
        super(GSATSRBP, self).__init__()
        assert n_hid_layers == len(hid_features) == len(n_heads), "Length of architecture parameters must match."

        # Initialization of parameters and layers
        self.n_drug, self.n_cir = n_drug, n_cir
        self.dropout_layer = nn.Dropout(dropout)

        # Convolution and GAT layers for feature extraction
        self.init_layers(n_in_features, hid_features, n_heads, residual)

        # Decoder for reconstructing interactions
        self.reconstructions = InnerProductDecoder(input_dim=hid_features[0], num_d=n_drug, act=torch.sigmoid)

        self.save_dict = {}
        self.ture_scores = None

    def init_layers(self, n_in_features, hid_features, n_heads, residual):
        """Initialize convolutional and GAT layers."""
        self.conv, self.conv_drug, self.conv_dis = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.gat_conv, self.gat_conv_drug, self.gat_conv_dis = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        sizes = [n_in_features] + hid_features
        for i in range(self.n_hid_layers):
            self.conv.append(SAGEConv(sizes[i], sizes[i + 1], normalize=False))
            self.conv_drug.append(SAGEConv(sizes[i], sizes[i + 1], normalize=False))
            self.conv_dis.append(SAGEConv(sizes[i], sizes[i + 1], normalize=False))
            self.gat_conv.append(GATConv(sizes[i + 1], sizes[i + 1], 1, residual=residual))
            self.gat_conv_drug.append(GATConv(sizes[i + 1], sizes[i + 1], 1, residual=residual))
            self.gat_conv_dis.append(GATConv(sizes[i + 1], sizes[i + 1], 1, residual=residual))

    def forward(self, x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir):
        """Forward pass for the model."""
        embd_heter, embd_drug, embd_cir = self.encode_features(x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir)
        final_embd = self.dropout_layer(embd_heter)
        return self.reconstructions(final_embd, self.dropout_layer(embd_cir), self.dropout_layer(embd_drug))

    def encode_features(self, x, edge_idx, x_drug, edge_idx_drug, x_cir, edge_idx_cir):
        """Encode features through convolutional and GAT layers."""
        # Process heterogeneous graph
        embd_heter = self.process_layer(x, edge_idx, self.conv, self.gat_conv)
        # Process drug graph
        embd_drug = self.process_layer(x_drug, edge_idx_drug, self.conv_drug, self.gat_conv_drug)
        # Process disease graph
        embd_cir = self.process_layer(x_cir, edge_idx_cir, self.conv_dis, self.gat_conv_dis)
        return embd_heter, embd_drug, embd_cir

    def process_layer(self, x, edge_idx, conv_layers, gat_layers):
        """Utility function to process layers."""
        embd_tmp = x
        for conv, gat in zip(conv_layers, gat_layers):
            embd_tmp = conv(embd_tmp, edge_idx)  # GraphSAGE Convolution
            embd_tmp = gat(embd_tmp, edge_idx)  # GAT Convolution
        return embd_tmp

    def save_func(self, data_file):
        """Save the model's data and exit the program."""
        data_concat = torch.cat((self.save_dict['ER'][self.save_dict['idx'][0]],
                                 self.save_dict['ED'][self.save_dict['idx'][1]]), dim=1).cpu().numpy()
        train_data_concat = data_concat[self.save_dict['y_train_idx']]
        test_data_concat = data_concat[self.save_dict['y_test_idx']]

        if not os.path.exists('../mid_data/'):
            os.makedirs('../mid_data/')

        scores = self.save_dict['ER'] @ self.save_dict['ED'].T

        joblib.dump({
            'x_train': train_data_concat,
            'x_test': test_data_concat,
            'y_train': self.save_dict['y_true'][self.save_dict['y_train_idx']].cpu().numpy(),
            'y_test': self.save_dict['y_true'][self.save_dict['y_test_idx']].cpu().numpy(),
            'scores': scores.cpu().numpy(),
            'true_scores': self.ture_scores,
        }, f'../mid_data/scores-{data_file}.dict')
        sys.exit(0)
