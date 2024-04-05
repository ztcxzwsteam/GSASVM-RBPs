import numpy as np
import pandas as pd
import os
import random
import torch
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from torch_geometric.utils import negative_sampling

def set_seed(seed):
    """Set the seed for reproducibility across multiple libraries."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def GIP_kernel(asso_matrix):
    """
    Gaussian Interaction Profile (GIP) kernel calculation for association matrix.
    """
    square_norm = np.square(np.linalg.norm(asso_matrix, axis=1))
    r = np.mean(square_norm)
    if r == 0:
        return np.zeros(asso_matrix.shape)

    norm_diff = np.square(np.linalg.norm(asso_matrix[:, None] - asso_matrix, axis=2))
    return np.exp(-norm_diff / r)

def get_syn_sim(A, seq_sim, str_sim, mode=0):
    """
    Compute synthetic similarity matrices for circRNA and diseases.
    """
    GIP_c_sim, GIP_d_sim = GIP_kernel(A), GIP_kernel(A.T)
    if mode == 0:
        return GIP_c_sim, GIP_d_sim

    # Compute the synthetic similarity matrix by averaging GIP and provided similarity matrices.
    syn_c = (np.where(seq_sim == 0, GIP_c_sim, seq_sim) + GIP_c_sim) / 2
    syn_d = (np.where(str_sim == 0, GIP_d_sim, str_sim) + GIP_d_sim) / 2
    return syn_c, syn_d

def k_matrix(matrix, k=20):
    """
    Retain the top-k similarities for each item in the matrix, ensuring the matrix remains symmetric.
    """
    idx = np.argsort(-matrix, axis=1)
    for i in range(matrix.shape[0]):
        matrix[i, idx[i, k:]] = 0
    return (matrix + matrix.T) / 2

def crossval_index(drug_mic_matrix, k_flod):
    """
    Create indices for k-fold cross-validation.
    """
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    # Ensure balanced positive and negative samples
    neg_index = np.random.choice(neg_index_matrix.shape[1], pos_index_matrix.shape[1], replace=False)
    neg_index_matrix = neg_index_matrix[:, neg_index]

    pos_index = random_index(pos_index_matrix, k_flod)
    neg_index = random_index(neg_index_matrix, k_flod)
    # Combine and shuffle indices for cross-validation
    index = [np.hstack((pos_index[i], neg_index[i])) for i in range(k_flod)]
    return [np.random.permutation(idx) for idx in index]

def random_index(index_matrix, k_fold):
    """
    Randomly shuffle indices for cross-validation splits.
    """
    total_indices = index_matrix.shape[1]
    shuffled_indices = np.random.permutation(index_matrix.T).tolist()
    split_size = total_indices // k_fold
    splits = [shuffled_indices[i*split_size : (i+1)*split_size] for i in range(k_fold)]
    # Add the remainder to the last split if needed
    if total_indices % k_fold != 0:
        splits[-1].extend(shuffled_indices[-(total_indices % k_fold):])
    return splits

def calculate_loss(pred, true_label):
    """
    Calculate binary cross-entropy loss.
    """
    loss_fun = torch.nn.BCELoss(reduction='mean')
    return loss_fun(pred, true_label)

def calculate_evaluation_metrics(pred_labels, true_labels):
    """
    Calculate AUC, AUPR, F1-score, accuracy, recall, specificity, and precision.
    """
    fpr, tpr, thresholds = roc_auc_score(true_labels, pred_labels)
    precision, recall, _ = precision_recall_curve(true_labels, pred_labels)
    AUC = auc(fpr, tpr)
    AUPR = auc(recall, precision)
    # Calculate other metrics based on optimal threshold determined by F1-score or another criterion
    # F1-score, accuracy, specificity, and precision calculation goes here
    return AUC, AUPR, f1_score, accuracy,
