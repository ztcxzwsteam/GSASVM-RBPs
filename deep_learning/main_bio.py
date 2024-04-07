import os
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from utils import set_seed, load_data, calculate_loss, calculate_evaluation_metrics, construct_adj_mat, construct_het_mat
from model import GSATSRBP as MNGACDA

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_data(temp_drug_cir, ablation=None):
    """Prepare the dataset including both positive and negative samples."""
    # Extract positive (existing links) and negative (non-existing links) samples
    pos_samples = np.argwhere(temp_drug_cir != 0)
    neg_samples = np.argwhere(temp_drug_cir == 0)

    # Adjust the number of negative samples based on the ablation study requirement
    ablation_factor = {'1-5': 5, '1-10': 10}.get(ablation, 1)
    selected_neg_samples = neg_samples[np.random.choice(len(neg_samples), len(pos_samples) * ablation_factor, replace=False)]

    # Combine and label the data
    data = np.vstack((pos_samples, selected_neg_samples))
    labels = np.hstack((np.ones(len(pos_samples)), np.zeros(len(selected_neg_samples))))

    return torch.tensor(data, dtype=torch.long), torch.tensor(labels, dtype=torch.float).unsqueeze(1)

def train_main(config, device):
    """Train and evaluate the model based on the provided configuration."""
    for data_name in config['data_list']:
        drug_sim, cir_sim, edge_idx_dict, drug_dis_matrix, true_labels = load_data(data_name)
        n_drug, n_cir = drug_sim.shape[0], cir_sim.shape[0]

        # Correct the diagonal of similarity matrices
        np.fill_diagonal(drug_sim, 0)
        np.fill_diagonal(cir_sim, 0)

        # Prepare the initial adjacency matrix and similarity matrices
        temp_drug_cir = np.zeros((n_drug, n_cir))
        temp_drug_cir[edge_idx_dict['pos_edges']] = 1  # Set positive edges

        # Model initialization and preparation
        model = MNGACDA(n_drug + n_cir, config['num_hidden_layers'], config['num_embedding_features'],
                        config['num_heads_per_layer'], n_drug, n_cir,
                        config['add_layer_attn'], config['residual']).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=5e-5, max_lr=config['lr'], step_size_up=200,
                                                step_size_down=200, mode='exp_range', gamma=0.99, cycle_momentum=False)

        # K-Fold Cross-Validation
        kf = KFold(n_splits=config['kfolds'], shuffle=True, random_state=123)
        metrics_list = []
        for train_idx, test_idx in kf.split(true_labels):
            y, edge_index_all = get_data(temp_drug_cir)
            # Training and evaluation logic...

        # Metrics processing and output for each dataset
        print(f'Completed training for {data_name}. Metrics: {np.mean(metrics_list, axis=0)}')

def train_deep():
    set_seed(666)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'kfolds': 5,
        'num_heads_per_layer': 3,
        'num_embedding_features': 128,
        'num_hidden_layers': 2,
        'num_epoch': 30,
        'knn_nums': 25,
        'lr': 1e-3,
        'weight_decay': 5e-3,
        'add_layer_attn': True,
        'residual': True,
        'data_list': ['human_m', 'mouse_m']
    }

    train_main(config, device)

    return 0

if __name__ == "__main__":
    train_deep
