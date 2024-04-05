import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.nn.inits import glorot

class GraphAttentionLayer(MessagePassing):
    def __init__(self, in_features, out_features, n_heads, residual=True, dropout=0.6, slope=0.2, activation=F.elu):
        # Initialize the MessagePassing class with 'mean' aggregation.
        super().__init__(aggr='mean')
        self.in_features = in_features  # Number of input features.
        self.out_features = out_features  # Number of output features.
        self.n_heads = n_heads  # Number of attention heads.
        self.residual = residual  # Whether to use a residual connection.

        # Dropout layers for features and attention coefficients.
        self.dropout = nn.Dropout(dropout)
        # LeakyReLU activation for the attention mechanism.
        self.leakyrelu = nn.LeakyReLU(slope)
        # Activation function to apply after aggregating the features.
        self.activation = activation

        # Weight matrix to transform input features.
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features * n_heads))
        # Attention vector to learn attention coefficients.
        self.attn = nn.Parameter(torch.Tensor(1, n_heads, out_features))
        # Optional projection layer for residual connections.
        self.proj = Linear(in_features, out_features * n_heads, bias=False) if residual else None

        # Initialize parameters.
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weight and attention vectors using the glorot method.
        glorot(self.weight)
        glorot(self.attn)
        # Reset parameters of the projection layer if it's used.
        if self.proj:
            self.proj.reset_parameters()

    def forward(self, x, edge_index, size=None):
        # Apply dropout to the input features.
        x = self.dropout(x)

        # Linearly transform input features and reshape for attention heads.
        x = torch.matmul(x, self.weight).view(-1, self.n_heads, self.out_features)
        # Propagate the transformed features through the graph.
        out = self.propagate(edge_index, x=x, size=size)

        # Apply residual connection if enabled.
        if self.residual:
            out += self.proj(x.view(-1, self.in_features))

        # Average the features across the attention heads.
        out = out.mean(dim=1)
        # Apply the activation function if specified.
        if self.activation:
            out = self.activation(out)

        return out


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim, dropout=0.0, act=torch.sigmoid):
        super().__init__()
        # Dropout layer for input embeddings.
        self.dropout = nn.Dropout(dropout)
        # Activation function to apply to the output.
        self.act = act

        # Linear layer to combine features from input embeddings.
        self.combination = nn.Linear(input_dim * 2, input_dim)
        # Learnable parameters to weight the importance of drug and circRNA embeddings.
        self.att_drug = nn.Parameter(torch.rand(2), requires_grad=True)
        self.att_cir = nn.Parameter(torch.rand(2), requires_grad=True)
        # Initialize the weights of the combination layer.
        nn.init.xavier_uniform_(self.combination.weight)

    def forward(self, inputs, embd_cir, embd_drug):
        # Check if inputs are provided, if not, use only the embeddings.
        if inputs is None:
            R, D = embd_drug, embd_cir
        else:
            # Apply dropout to inputs and embeddings.
            inputs, embd_drug, embd_cir = map(self.dropout, (inputs, embd_drug, embd_cir))
            # Split inputs for drugs and circRNAs, and concatenate with respective embeddings.
            num_d = inputs.size(0) // 2
            R, D = inputs[:num_d], inputs[num_d:]
            R = torch.cat((R, embd_drug), dim=1)
            D = torch.cat((D, embd_cir), dim=1)

        # Apply linear transformation to combined features.
        R, D = self.combination(R), self.combination(D).T
        # Compute the pairwise inner product and apply the activation function.
        outputs = self.act(torch.flatten(R @ D))
        return outputs
