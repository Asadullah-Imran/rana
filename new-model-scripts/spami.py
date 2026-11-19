#!/usr/bin/env python3
"""
SpaMI Multimodal Spatial Clustering Benchmark Script
Faithful reproduction of official SpaMI (https://pmc.ncbi.nlm.nih.gov/articles/PMC12503342)
from source-code/SpaMI-main/, wrapped in the multi-dataset, multi-seed 8-metric benchmark suite.
"""

import argparse
import os
import random
import shutil
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import seaborn as sns
import sklearn
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.utils.extmath
import torch
import torch.nn as nn
import torch.nn.functional as F
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.backends import cudnn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    import episcanpy
    HAS_EPISCANPY = True
except ImportError:
    HAS_EPISCANPY = False


# ================================================================================================
# 1. UTILS (from source-code/SpaMI-main/SpaMI/utils.py)
# ================================================================================================

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# ================================================================================================
# 2. PREPROCESSING (from source-code/SpaMI-main/SpaMI/preprocess.py & Tutorials 1-4)
# ================================================================================================

def tfidf(X):
    # TF-IDF normalization (official preprocess.py:139-148)
    col_sums = np.asarray(X.sum(axis=0)).ravel()
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    idf = X.shape[0] / col_sums

    if scipy.sparse.issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        tf = X.multiply(1.0 / row_sums[:, None])
        return tf.multiply(idf)
    else:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        tf = X / row_sums
        return tf * idf


def seurat_clr(x):
    # Centered log-ratio normalization (official preprocess.py:113-117)
    s = np.sum(np.log1p(x[x > 0]))
    exp = np.exp(s / len(x))
    return np.log1p(x / exp)


def construct_adj(adata, n_neighbors=3):
    # Spatial neighbor graph construction (official preprocess.py:151-169)
    coordinate = np.asarray(adata.obsm['spatial'], dtype=np.float64)
    if HAS_OT:
        distance_matrix = ot.dist(coordinate, coordinate, metric='euclidean')
    else:
        distance_matrix = cdist(coordinate, coordinate, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    graph_neigh = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    return adj, graph_neigh


def permutation(feature):
    # Feature permutation for contrastive learning (official preprocess.py:172-177)
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    return feature_permutated


def add_contrastive_label(adata):
    # Contrastive learning labels (official preprocess.py:180-186)
    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    return label_CSL


def preprocess_dataset(adata_omics1, adata_omics2, dataset_name, n_neighbors=3):
    """
    Dataset-specific preprocessing faithfully matching official SpaMI Tutorials & preprocess.py:
      - Mouse Brain (MISAR-seq RNA + ATAC): matches Tutorial_1.ipynb (dataset='MISAR')
      - 10x Human Lymph Node (RNA + ADT): matches Tutorial_3.ipynb & preprocess.py
    """
    if dataset_name.startswith("10x"):
        # ---------- ADT / Protein preprocessing (Tutorial_3.ipynb Cell 4 & preprocess.py:111-123) ----------
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        sc.pp.filter_genes(adata_omics2, min_cells=50)
        X_adt = adata_omics2.X.toarray() if scipy.sparse.issparse(adata_omics2.X) else np.array(adata_omics2.X)
        adata_omics2.X = np.apply_along_axis(seurat_clr, 1, X_adt)
        feat_omics2 = np.ascontiguousarray(adata_omics2.X)

        # ---------- RNA preprocessing (Tutorial_3.ipynb Cell 5) ----------
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=3000)
        adata_omics1_copy = adata_omics1[:, adata_omics1.var['highly_variable']]
        if scipy.sparse.issparse(adata_omics1_copy.X):
            feat_omics1 = adata_omics1_copy.X.toarray()[:, ]
        else:
            feat_omics1 = adata_omics1_copy.X[:, ]
        n_pca_comps = min(adata_omics2.n_vars, feat_omics1.shape[1], feat_omics1.shape[0] - 1)
        pca = sklearn.decomposition.PCA(n_components=n_pca_comps)
        feat_omics1 = np.ascontiguousarray(pca.fit_transform(feat_omics1))

    else:
        # ---------- Mouse Brain MISAR-seq RNA preprocessing (Tutorial_1.ipynb Cell 10) ----------
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.highly_variable_genes(adata_omics1, n_top_genes=1800)
        adata_omics1_copy = adata_omics1[:, adata_omics1.var['highly_variable']]
        if scipy.sparse.issparse(adata_omics1_copy.X):
            feat_omics1 = adata_omics1_copy.X.toarray()[:, ]
        else:
            feat_omics1 = adata_omics1_copy.X[:, ]
        pca = sklearn.decomposition.PCA(n_components=50)
        feat_omics1 = np.ascontiguousarray(pca.fit_transform(feat_omics1))

        # ---------- Mouse Brain MISAR-seq ATAC preprocessing (Tutorial_1.ipynb Cell 11 & preprocess.py:77-92) ----------
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        adata_omics2_copy = adata_omics2.copy()
        min_cells_atac = int(np.ceil(0.005 * adata_omics2.shape[0]))
        if HAS_EPISCANPY:
            episcanpy.pp.binarize(adata_omics2_copy)
            episcanpy.pp.filter_features(adata_omics2_copy, min_cells=min_cells_atac)
        else:
            if scipy.sparse.issparse(adata_omics2_copy.X):
                X_bin = adata_omics2_copy.X.tocsr().copy()
                X_bin.data = (X_bin.data > 0).astype(np.float32)
                adata_omics2_copy.X = X_bin
            else:
                adata_omics2_copy.X = (np.asarray(adata_omics2_copy.X) > 0).astype(np.float32)
            sc.pp.filter_genes(adata_omics2_copy, min_cells=min_cells_atac)

        count_mat = adata_omics2_copy.X.copy()
        X = tfidf(count_mat)
        X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, 51)[0]
        X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        X_lsi_std = X_lsi.std(axis=1, ddof=1, keepdims=True)
        X_lsi_std = np.where(X_lsi_std == 0, 1.0, X_lsi_std)
        X_lsi /= X_lsi_std
        X_lsi = X_lsi[:, 1:]
        if scipy.sparse.issparse(X_lsi):
            feat_omics2 = np.ascontiguousarray(X_lsi.toarray()[:, ])
        else:
            feat_omics2 = np.ascontiguousarray(X_lsi[:, ])

    # Construct spatial neighbor graphs and contrastive labels
    adj_omics1, graph_neigh_omics1 = construct_adj(adata_omics1, n_neighbors=n_neighbors)
    label_CSL_omics1 = add_contrastive_label(adata_omics1)

    adj_omics2, graph_neigh_omics2 = construct_adj(adata_omics2, n_neighbors=n_neighbors)
    label_CSL_omics2 = add_contrastive_label(adata_omics2)

    omics1_data = {
        'feat': feat_omics1,
        'adj': adj_omics1,
        'graph_neigh': graph_neigh_omics1,
        'label_CSL': label_CSL_omics1,
        'spatial_locations': adata_omics1.obsm['spatial'],
    }
    omics2_data = {
        'feat': feat_omics2,
        'adj': adj_omics2,
        'graph_neigh': graph_neigh_omics2,
        'label_CSL': label_CSL_omics2,
    }
    return omics1_data, omics2_data


# ================================================================================================
# 3. MODEL ARCHITECTURE (exact from source-code/SpaMI-main/SpaMI/model.py)
# ================================================================================================

class SpaMI(nn.Module):
    def __init__(self, omics1_input_dim, omics2_input_dim, out_dim, dropout_rate,
                 omics1_feat, omics1_adj, omics1_graph_neigh, omics1_label_CSL,
                 omics2_feat, omics2_adj, omics2_graph_neigh, omics2_label_CSL):
        super(SpaMI, self).__init__()

        # dimensions of input and output
        self.omics1_input_dim = omics1_input_dim
        self.omics2_input_dim = omics2_input_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        # omics1 data
        self.omics1_feat = omics1_feat
        self.omics1_adj = omics1_adj
        self.omics1_graph_neigh = omics1_graph_neigh
        self.omics1_label_CSL = omics1_label_CSL

        # omics2 data
        self.omics2_feat = omics2_feat
        self.omics2_adj = omics2_adj
        self.omics2_graph_neigh = omics2_graph_neigh
        self.omics2_label_CSL = omics2_label_CSL

        # encoder
        self.omics1_encoder = omics1_encoder(self.omics1_input_dim, self.out_dim, self.dropout_rate)
        self.omics2_encoder = omics2_encoder(self.omics2_input_dim, self.out_dim, self.dropout_rate)

        # decoder
        self.omics1_decoder = omics1_decoder(self.omics1_input_dim, self.out_dim, self.dropout_rate)
        self.omics2_decoder = omics2_decoder(self.omics2_input_dim, self.out_dim, self.dropout_rate)

        # attention aggregation layer
        self.attention_layer = AttentionLayer(out_dim, out_dim)

    def forward(self, omics1_feat_shuffle, omics2_feat_shuffle):
        omics1_emb, omics1_ret, omics1_ret_a = self.omics1_encoder(
            self.omics1_feat, omics1_feat_shuffle,
            self.omics1_adj, self.omics1_graph_neigh
        )

        omics2_emb, omics2_ret, omics2_ret_a = self.omics2_encoder(
            self.omics2_feat, omics2_feat_shuffle,
            self.omics2_adj, self.omics2_graph_neigh
        )

        combine_emb, omics_weight = self.attention_layer(omics1_emb, omics2_emb)

        omics1_rec = self.omics1_decoder(combine_emb, self.omics1_adj)
        omics2_rec = self.omics2_decoder(combine_emb, self.omics2_adj)

        return (omics1_emb, omics1_rec, omics1_ret, omics1_ret_a,
                omics2_emb, omics2_rec, omics2_ret, omics2_ret_a,
                combine_emb, omics_weight)


class omics1_encoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics1_encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_conv1 = GCNConv(input_dim, 256)
        self.encoder_conv2 = GCNConv(256, out_dim)

        self.discriminator = Discriminator(out_dim)
        self.readout = AvgReadout()

    def forward(self, feat, feat_a, adj, graph_neigh):
        edge_index, _ = dense_to_sparse(adj)

        z = self.dropout(feat)
        z = self.encoder_conv1(z, edge_index)
        z = F.relu(z)
        z = self.encoder_conv2(z, edge_index)
        emb = z

        z_a = self.dropout(feat_a)
        z_a = self.encoder_conv1(z_a, edge_index)
        z_a = F.relu(z_a)
        z_a = self.encoder_conv2(z_a, edge_index)
        emb_a = z_a

        # Local neighbor representation
        g = self.readout(emb, graph_neigh)
        g = torch.sigmoid(g)

        # Corrupted local neighbor representation
        g_a = self.readout(emb_a, graph_neigh)
        g_a = torch.sigmoid(g_a)

        # discriminator for contrastive learning
        ret = self.discriminator(g, emb, emb_a)
        ret_a = self.discriminator(g_a, emb_a, emb)

        return emb, ret, ret_a


class omics1_decoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics1_decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_conv1 = GCNConv(out_dim, 256)
        self.decoder_conv2 = GCNConv(256, input_dim)

    def forward(self, emb, adj):
        edge_index, _ = dense_to_sparse(adj)

        h = self.dropout(emb)
        h = self.decoder_conv1(h, edge_index)
        h = F.relu(h)
        h = self.decoder_conv2(h, edge_index)
        omics1_rec = h

        return omics1_rec


class omics2_encoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics2_encoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.encoder_conv1 = GCNConv(input_dim, 256)
        self.encoder_conv2 = GCNConv(256, out_dim)

        self.discriminator = Discriminator(out_dim)
        self.readout = AvgReadout()

    def forward(self, feat, feat_a, adj, graph_neigh):
        edge_index, _ = dense_to_sparse(adj)

        z = self.dropout(feat)
        z = self.encoder_conv1(z, edge_index)
        z = F.relu(z)
        z = self.encoder_conv2(z, edge_index)
        emb = z

        z_a = self.dropout(feat_a)
        z_a = self.encoder_conv1(z_a, edge_index)
        z_a = F.relu(z_a)
        z_a = self.encoder_conv2(z_a, edge_index)
        emb_a = z_a

        # Local neighbor representation
        g = self.readout(emb, graph_neigh)
        g = torch.sigmoid(g)

        # Corrupted local neighbor representation
        g_a = self.readout(emb_a, graph_neigh)
        g_a = torch.sigmoid(g_a)

        ret = self.discriminator(g, emb, emb_a)
        ret_a = self.discriminator(g_a, emb_a, emb)

        return emb, ret, ret_a


class omics2_decoder(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate):
        super(omics2_decoder, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.decoder_conv1 = GCNConv(out_dim, 256)
        self.decoder_conv2 = GCNConv(256, input_dim)

    def forward(self, emb, adj):
        edge_index, _ = dense_to_sparse(adj)

        h = self.dropout(emb)
        h = self.decoder_conv1(h, edge_index)
        h = F.relu(h)
        h = self.decoder_conv2(h, edge_index)
        omics2_rec = h

        return omics2_rec


class AttentionLayer(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(AttentionLayer, self).__init__()
        self.w_omega = nn.Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = nn.Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = torch.cat((torch.unsqueeze(emb1, 1), torch.unsqueeze(emb2, 1)), 1)
        v = torch.tanh(torch.matmul(emb, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(torch.squeeze(vu) + 1e-6, dim=1)
        emb_combined = torch.matmul(torch.transpose(emb, 1, 2), torch.unsqueeze(alpha, -1))
        return torch.squeeze(emb_combined), alpha


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


# ================================================================================================
# 4. TRAINING (exact from source-code/SpaMI-main/SpaMI/main.py)
# ================================================================================================

def train_spami(omics1_data, omics2_data, dataset, device,
                out_dim=64, dropout_rate=0.15, weight_decay=0.0, random_seed=2024):
    fix_seed(random_seed)

    # omics1 data
    omics1_feat = torch.FloatTensor(np.ascontiguousarray(omics1_data['feat'], dtype=np.float32)).to(device)
    omics1_feat_shuffle = None
    omics1_label_CSL = torch.FloatTensor(np.ascontiguousarray(omics1_data['label_CSL'], dtype=np.float32)).to(device)
    omics1_graph_neigh = torch.FloatTensor(
        np.ascontiguousarray(omics1_data['graph_neigh'] + np.eye(omics1_data['adj'].shape[0]), dtype=np.float32)
    ).to(device)
    omics1_input_dim = omics1_data['feat'].shape[1]
    omics1_adj = torch.FloatTensor(
        np.ascontiguousarray(omics1_data['adj'] + np.eye(omics1_data['adj'].shape[0]), dtype=np.float32)
    ).to(device)

    # omics2 data
    omics2_feat = torch.FloatTensor(np.ascontiguousarray(omics2_data['feat'], dtype=np.float32)).to(device)
    omics2_feat_shuffle = None
    omics2_label_CSL = torch.FloatTensor(np.ascontiguousarray(omics2_data['label_CSL'], dtype=np.float32)).to(device)
    omics2_graph_neigh = torch.FloatTensor(
        np.ascontiguousarray(omics2_data['graph_neigh'] + np.eye(omics2_data['adj'].shape[0]), dtype=np.float32)
    ).to(device)
    omics2_input_dim = omics2_data['feat'].shape[1]
    omics2_adj = torch.FloatTensor(
        np.ascontiguousarray(omics2_data['adj'] + np.eye(omics2_data['adj'].shape[0]), dtype=np.float32)
    ).to(device)

    # parameters for different datasets (exact from official main.py:34-60)
    if dataset == 'MISAR':
        learning_rate = 0.01
        epochs = 800
        factors = [1, 1, 10, 15, 3]
    elif dataset == 'Mouse_Brain_P22':
        learning_rate = 0.001
        epochs = 400
        factors = [1, 1, 10, 15, 5]
    elif dataset == "Mouse_Thymus":
        learning_rate = 0.0001
        epochs = 600
        factors = [1, 5, 10, 25, 1]
    else:
        learning_rate = 0.001
        epochs = 1000
        factors = [1, 1, 10, 10, 5]

    # model
    model = SpaMI(
        omics1_input_dim, omics2_input_dim, out_dim, dropout_rate,
        omics1_feat, omics1_adj, omics1_graph_neigh, omics1_label_CSL,
        omics2_feat, omics2_adj, omics2_graph_neigh, omics2_label_CSL
    ).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    model.train()

    loss_rec = nn.MSELoss()
    loss_CSL = nn.BCEWithLogitsLoss()

    train_losses = []

    # train
    for epoch in tqdm(range(epochs)):
        model.train()

        omics1_feat_shuffle = permutation(omics1_feat)
        omics2_feat_shuffle = permutation(omics2_feat)

        (omics1_emb, omics1_rec, omics1_ret, omics1_ret_a,
         omics2_emb, omics2_rec, omics2_ret, omics2_ret_a,
         combine_emb, omics_weight) = model(omics1_feat_shuffle, omics2_feat_shuffle)

        # contrastive learning loss
        loss_omics1_csl1 = loss_CSL(omics1_ret, omics1_label_CSL)
        loss_omics1_csl2 = loss_CSL(omics1_ret_a, omics1_label_CSL)
        loss_omics1_CSL = loss_omics1_csl1 + loss_omics1_csl2

        loss_omics2_csl1 = loss_CSL(omics2_ret, omics2_label_CSL)
        loss_omics2_csl2 = loss_CSL(omics2_ret_a, omics2_label_CSL)
        loss_omics2_CSL = loss_omics2_csl1 + loss_omics2_csl2

        # reconstruction loss
        loss_omics1_rec = loss_rec(omics1_feat, omics1_rec)
        loss_omics2_rec = loss_rec(omics2_feat, omics2_rec)

        # cosine similarity loss
        cosine_sim = F.cosine_similarity(omics1_emb, omics2_emb, dim=1)
        loss_cosine = 1 - cosine_sim.mean()

        loss = (factors[0] * loss_omics1_CSL + factors[1] * loss_omics2_CSL +
                factors[2] * loss_omics1_rec + factors[3] * loss_omics2_rec +
                factors[4] * loss_cosine)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    print('Model training completed!')

    with torch.no_grad():
        model.eval()
        (omics1_emb, omics1_rec, _, _,
         omics2_emb, omics2_rec, _, _,
         combine_emb, omics_weight) = model(omics1_feat_shuffle, omics2_feat_shuffle)

        output = {
            'emb_omics1': omics1_emb.clone().detach().cpu().numpy(),
            'emb_omics2': omics2_emb.clone().detach().cpu().numpy(),
            'rec_omics1': omics1_rec.clone().detach().cpu().numpy(),
            'rec_omics2': omics2_rec.clone().detach().cpu().numpy(),
            'combine_emb': combine_emb.clone().detach().cpu().numpy(),
            'omics_weight': omics_weight.clone().detach().cpu().numpy(),
            'train_losses': train_losses,
        }

    return output


# ================================================================================================
# 5. CLUSTERING (Leiden clustering matching official utils.py & Tutorials 1-4)
# ================================================================================================

def search_leiden_res(adata, n_clusters, use_rep='X', n_neighbors=50,
                      start=0.1, end=3.0, increment=0.05, random_state=0):
    """
    Search Leiden resolution to match target number of ground-truth clusters,
    using official SpaMI neighbor graph settings (n_neighbors=50).
    """
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep)
    best_res = start
    best_diff = float('inf')

    for res in np.arange(start, end + 1e-9, increment):
        sc.tl.leiden(adata, random_state=random_state, resolution=float(res))
        n_found = adata.obs['leiden'].nunique()
        diff = abs(n_found - n_clusters)
        if diff < best_diff:
            best_diff = diff
            best_res = float(res)
        if n_found == n_clusters:
            return float(res)

    return best_res


def cluster_spami(adata, n_clusters, use_pca=False, n_comps=20, n_neighbors=50, random_state=0):
    """
    Perform Leiden clustering on SpaMI joint embedding (matching Tutorial_1 & Tutorial_3).
      - MISAR (out_dim=20): use_pca=False, use_rep='SpaMI' (matches Tutorial_1.ipynb Cell 20)
      - 10x HLN (out_dim=64): use_pca=True (n_comps=20), use_rep='X_pca' (matches Tutorial_3.ipynb Cell 20 & utils.py)
    """
    if use_pca:
        emb_adata = AnnData(adata.obsm['SpaMI'].copy())
        sc.pp.pca(emb_adata, n_comps=min(n_comps, adata.obsm['SpaMI'].shape[1]))
        adata.obsm['SpaMI_pca'] = emb_adata.obsm['X_pca']
        use_rep = 'SpaMI_pca'
    else:
        use_rep = 'SpaMI'

    res = search_leiden_res(
        adata,
        n_clusters=n_clusters,
        use_rep=use_rep,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    sc.tl.leiden(adata, random_state=random_state, resolution=res)
    adata.obs['SpaMI'] = adata.obs['leiden'].astype(str).astype('category')
    return adata, res


# ================================================================================================
# 6. ARGUMENT PARSING & BENCHMARK CONFIGURATION (matching new-scripts/)
# ================================================================================================

parser = argparse.ArgumentParser(description="Run SpaMI multimodal spatial clustering benchmark")
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--dropout_rate', type=float, default=0.15)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dataset', type=int, default=None, help="Single dataset index to run (0 to 5)")
parser.add_argument('--dataset_indices', type=int, nargs='+', default=None, help="List of dataset indices (e.g. --dataset_indices 0 1)")
parser.add_argument('--seed', type=int, default=None, help="Single seed to run (e.g. --seed 2024)")
parser.add_argument('--seeds', type=int, nargs='+', default=None, help="List of seeds to run (e.g. --seeds 42 0 7)")
parser.add_argument('--output_dir', type=str, default=None, help="Custom output directory")

args, unknown = parser.parse_known_args()
device = torch.device(args.device)

print(f"\n[Hardware] Compute Engine: {device}")
if torch.cuda.is_available() and device.type == 'cuda':
    print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
else:
    print("[Hardware] No GPU detected or CPU selected. Running on CPU.\n")

choices = [
    ("10x_human_lymph_node_A1",
     "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1",
     "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1",
     "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1",
     "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1",
     "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1",
     "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

if args.dataset is not None:
    DATASET_INDICES = [args.dataset]
elif args.dataset_indices is not None:
    DATASET_INDICES = args.dataset_indices
else:
    DATASET_INDICES = [0, 1, 2, 3, 4, 5]

if args.seed is not None:
    SEEDS = [args.seed]
elif args.seeds is not None:
    SEEDS = args.seeds
else:
    SEEDS = [42, 0, 7, 123, 1234, 2022, 2023, 2024, 1337]

MODEL_NAME = "SpaMI"
OUTPUT_DIR = args.output_dir if args.output_dir else os.getenv("OUTPUT_DIR", f"{MODEL_NAME}_outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALS_DIR = os.path.join(OUTPUT_DIR, "visuals")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)

all_results = []
all_dataset_adata = {}

for dataset_idx in DATASET_INDICES:
    dataset_name, folder_url = choices[dataset_idx]

    print("\n" + "#" * 80)
    print(f" STARTING DATASET: {dataset_name} ".center(80, "#"))
    print("#" * 80)

    # ------------------ Download & Path Configuration ------------------
    base = f"data/{dataset_name}"
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")

    if dataset_name.startswith("10x"):
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        spami_dataset_key = "10x_HLN"  # uses official `else:` branch in main.py
        spami_out_dim = 64
        spami_n_neighbors = 3
        cluster_use_pca = True
    else:
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        spami_dataset_key = "MISAR"    # uses official `dataset == 'MISAR'` branch in main.py & Tutorial_1
        spami_out_dim = 20
        spami_n_neighbors = 3
        cluster_use_pca = False

    if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
        print(f"Downloading dataset files into: {base}")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{folder_url}" --output "{base}" --remaining-ok')
        for root, dirs, files in os.walk(base):
            if root != base:
                for f in files:
                    src = os.path.join(root, f)
                    dst = os.path.join(base, f)
                    if not os.path.exists(dst):
                        shutil.move(src, dst)
    else:
        print(f"Dataset files already exist at {base}. Skipping download.")

    # ------------------ Load AnnData & Annotations ------------------
    adata_omics1 = sc.read_h5ad(rna_path)
    adata_omics2 = sc.read_h5ad(other_path)

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    adata_omics1.obs['ground_truth'] = anno_df[gt_column]
    adata_omics2.obs['ground_truth'] = anno_df[gt_column]

    # ------------------ Preprocessing ------------------
    fix_seed(2024)
    omics1_data, omics2_data = preprocess_dataset(
        adata_omics1, adata_omics2, dataset_name, n_neighbors=spami_n_neighbors
    )

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = int(adata_omics1.obs["ground_truth"].dropna().nunique())
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = n_ground_truth

    best_ari = -1.0
    best_seed = None
    best_adata = None

    for seed in SEEDS:
        print("\n" + "-" * 60)
        print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
        print("-" * 60)

        seed_start_time = time.time()

        # 1. Model Training
        output = train_spami(
            omics1_data=omics1_data,
            omics2_data=omics2_data,
            dataset=spami_dataset_key,
            device=device,
            out_dim=spami_out_dim,
            dropout_rate=args.dropout_rate,
            weight_decay=args.weight_decay,
            random_seed=seed,
        )

        # 2. Latent Representation & Clustering
        adata = adata_omics1.copy()
        adata.obsm['SpaMI'] = output['combine_emb'].copy()
        adata.obsm['emb_omics1'] = output['emb_omics1'].copy()
        adata.obsm['emb_omics2'] = output['emb_omics2'].copy()
        adata.obsm['omics_weight'] = output['omics_weight'].copy()

        adata, chosen_res = cluster_spami(
            adata,
            n_clusters=n_cluster,
            use_pca=cluster_use_pca,
            n_comps=20,
            n_neighbors=50,
            random_state=0,
        )

        duration_sec = round(time.time() - seed_start_time, 2)

        # 3. Score Generating (8 metrics + Duration, masked by valid_mask)
        if 'anno' not in adata.obs.columns and 'ground_truth' in adata.obs.columns:
            adata.obs['anno'] = adata.obs['ground_truth']

        valid_mask = adata.obs['ground_truth'].notna()
        y_true = adata.obs.loc[valid_mask, 'ground_truth'].astype(str)
        y_pred_eval = adata.obs.loc[valid_mask, 'SpaMI'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred_eval)
        nmi = normalized_mutual_info_score(y_true, y_pred_eval)
        ami = adjusted_mutual_info_score(y_true, y_pred_eval)
        homogeneity = homogeneity_score(y_true, y_pred_eval)
        v_measure = v_measure_score(y_true, y_pred_eval)

        joint_feat = adata.obsm['SpaMI'][valid_mask.values]
        le = LabelEncoder()
        y_pred_int = le.fit_transform(y_pred_eval)

        try:
            sil_score = float(silhouette_score(joint_feat, y_pred_int))
        except Exception:
            sil_score = float('nan')

        try:
            chi_score = float(calinski_harabasz_score(joint_feat, y_pred_int))
        except Exception:
            chi_score = float('nan')

        try:
            dbi_score = float(davies_bouldin_score(joint_feat, y_pred_int))
        except Exception:
            dbi_score = float('nan')

        print(f"\nResult for dataset: {dataset_name} | seed: {seed} (Leiden res={chosen_res:.2f})")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil_score:.4f} | CHI: {chi_score:.2f} | DBI: {dbi_score:.4f} | Duration: {duration_sec:.2f}s")

        res_dict = {
            'dataset': dataset_name,
            'seed': seed,
            'ARI': ari,
            'NMI': nmi,
            'AMI': ami,
            'Homogeneity': homogeneity,
            'V-measure': v_measure,
            'Silhouette': sil_score,
            'CHI': chi_score,
            'DBI': dbi_score,
            'Duration': duration_sec,
        }
        dataset_results.append(res_dict)
        all_results.append(res_dict)

        # Track best seed AnnData for cross-dataset visualizations
        if ari > best_ari or best_adata is None:
            best_ari = ari
            best_seed = seed
            best_adata = adata.copy()
            best_adata.obsm[f'{MODEL_NAME}_emb'] = best_adata.obsm['SpaMI'].copy()
            best_adata.obs[f'{MODEL_NAME}_domain'] = best_adata.obs['SpaMI'].copy()
            best_adata.obs['domain'] = best_adata.obs['SpaMI'].copy()
            best_adata.uns['selected_seed'] = best_seed

        # ------------------ Plotting & Visualization ------------------
        fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
        sc.pp.neighbors(adata, use_rep='SpaMI', n_neighbors=10)
        sc.tl.umap(adata)

        sc.pl.umap(adata, color='SpaMI', ax=ax_list[0], title='SpaMI', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='SpaMI', ax=ax_list[1], title='SpaMI', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='anno', ax=ax_list[2], title='anno', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        os.makedirs("results", exist_ok=True)
        plot_filename = os.path.join(VISUALS_DIR, f"{dataset_name}_seed_{seed}.png")
        plt.savefig(plot_filename, dpi=150)
        plt.savefig(f"results/{MODEL_NAME}_{dataset_name}_seed_{seed}.png", dpi=150)
        plt.close()

    # Save best-ARI seed AnnData for cross-dataset visualizations
    if best_adata is not None:
        all_dataset_adata[dataset_name] = best_adata
        best_adata_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_adata_{dataset_name}.h5ad")
        try:
            best_adata.write_h5ad(best_adata_path)
            print(f"[Saved] Best-ARI seed ({best_seed}) AnnData saved to: {best_adata_path}")
        except Exception as e:
            print(f"[Warn] Could not save AnnData for {dataset_name}: {e}")

    # ------------------ Dataset Summary & Boxplots ------------------
    df_ds = pd.DataFrame(dataset_results)
    print("\n" + "=" * 80)
    print(f" RESULTS SUMMARY FOR {dataset_name} ".center(80, "="))
    print("=" * 80)
    print(df_ds.to_string(index=False))
    print("-" * 80)
    print("Summary Statistics:")
    print(df_ds.describe().loc[['mean', 'std']])
    print("=" * 80)

    summary_lines = [f"Summary Stats for {dataset_name}:"]
    metrics_list = ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette', 'CHI', 'DBI', 'Duration']
    for metric in metrics_list:
        mean_val = df_ds[metric].mean()
        std_val = df_ds[metric].std()
        summary_lines.append(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    dataset_summary_str = "\n".join(summary_lines)
    print(dataset_summary_str)
    print("=" * 80)

    with open(os.path.join(RESULTS_DIR, f"summary_{dataset_name}.txt"), "w") as f:
        f.write(dataset_summary_str)

    df_ds.to_csv(os.path.join(RESULTS_DIR, f"metrics_{dataset_name}.csv"), index=False)
    os.makedirs("results", exist_ok=True)
    df_ds.to_csv(f"results/{MODEL_NAME}_{dataset_name}_results.csv", index=False)
    df_ds.to_csv(f"results/metrics_{dataset_name}.csv", index=False)
    with open(f"results/summary_{dataset_name}.txt", "w") as f:
        f.write(dataset_summary_str)
    print(f"Saved dataset metrics to '{RESULTS_DIR}/metrics_{dataset_name}.csv'")
    print(f"Saved dataset summary to '{RESULTS_DIR}/summary_{dataset_name}.txt'")

    # Publication-quality boxplot + stripplot for this dataset
    plt.figure(figsize=(14, 6))
    df_ds_long = df_ds.melt(
        id_vars=['seed'],
        value_vars=['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette'],
        var_name='Metric',
        value_name='Score',
    )
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        palette='Set2',
        showfliers=False,
        width=0.5,
    )
    sns.stripplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        color='black',
        alpha=0.6,
        jitter=0.2,
        size=5,
        ax=ax,
    )
    plt.title(f"Variability of Metrics across {len(SEEDS)} Seeds ({dataset_name})", fontsize=14, pad=15)
    plt.xlabel("Metric", fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    boxplot_path = os.path.join(VISUALS_DIR, f"boxplot_{dataset_name}.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.savefig(f"results/boxplot_{dataset_name}.png", dpi=300)
    plt.close()
    print(f"Saved variability boxplot to '{boxplot_path}'\n")


# ================================================================================================
# 7. GLOBAL SUMMARY & PUBLICATION VISUALIZATION SUITE
# ================================================================================================

df_all = pd.DataFrame(all_results)
df_all.to_csv(os.path.join(RESULTS_DIR, f"{MODEL_NAME}_benchmark_results.csv"), index=False)
df_all.to_csv(os.path.join(RESULTS_DIR, "metrics_all_datasets.csv"), index=False)
df_all.to_csv(f"results/{MODEL_NAME}_all_results.csv", index=False)
df_all.to_csv("results/metrics_all_datasets.csv", index=False)

global_summary_lines = [
    "=" * 80,
    f"GLOBAL SUMMARY: MEAN ± STD ACROSS {len(SEEDS)} SEEDS ({MODEL_NAME})",
    "=" * 80,
]

for d_name in df_all['dataset'].unique():
    df_d = df_all[df_all['dataset'] == d_name]
    global_summary_lines.append(f"\nDataset: {d_name}")
    global_summary_lines.append("-" * len(f"Dataset: {d_name}"))
    for metric in ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette', 'CHI', 'DBI', 'Duration']:
        mean_val = df_d[metric].mean()
        std_val = df_d[metric].std()
        global_summary_lines.append(f"{metric} = {mean_val:.4f} ± {std_val:.4f}")
global_summary_lines.append("\n" + "=" * 80)

global_summary_str = "\n".join(global_summary_lines)
print(global_summary_str)

summary_stats_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_summary_statistics.txt")
with open(summary_stats_path, "w") as f:
    f.write(global_summary_str)
with open("results/summary_statistics.txt", "w") as f:
    f.write(global_summary_str)

print("\n" + "=" * 80)
print(" ALL DATASETS COMPLETED ".center(80, "="))
print("=" * 80)
print(df_all.groupby('dataset')[['ARI', 'NMI', 'Silhouette', 'CHI', 'DBI', 'Duration']].mean())
print("=" * 80)
print(f"Saved combined metrics to '{RESULTS_DIR}/{MODEL_NAME}_benchmark_results.csv'")
print(f"Saved global summary statistics to '{summary_stats_path}'")

# Comparative global boxplot across all datasets
df_all_long = df_all.melt(
    id_vars=['dataset', 'seed'],
    value_vars=['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette'],
    var_name='Metric',
    value_name='Score',
)

fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=True)
axes = axes.flatten()
metrics_list = ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette']

sns.set_theme(style="whitegrid")
for i, metric in enumerate(metrics_list):
    df_metric = df_all_long[df_all_long['Metric'] == metric]
    sns.boxplot(
        ax=axes[i],
        x='dataset',
        y='Score',
        data=df_metric,
        palette='Set3',
        showfliers=False,
    )
    sns.stripplot(
        ax=axes[i],
        x='dataset',
        y='Score',
        data=df_metric,
        color='black',
        alpha=0.5,
        jitter=0.15,
        size=4,
    )
    axes[i].set_title(f"{metric} Comparison", fontsize=14)
    axes[i].set_xlabel("Dataset", fontsize=12)
    axes[i].set_ylabel("Score", fontsize=12)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_ylim(0, 1.05)

plt.suptitle(f"{MODEL_NAME}: Performance Comparison Across All Datasets & {len(SEEDS)} Seeds", fontsize=18, y=0.98)
plt.tight_layout()
global_plot_path = os.path.join(VISUALS_DIR, f"{MODEL_NAME}_global_boxplot_comparison.png")
plt.savefig(global_plot_path, dpi=300)
plt.savefig("results/global_boxplot_comparison.png", dpi=300)
plt.close()
print(f"Saved global comparison boxplot to '{global_plot_path}'")

# Execute Full Visualization Suite
try:
    from visualizations import run_all_visualizations
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(script_dir)
    sys.path.append(os.path.join(os.path.dirname(script_dir), "new-scripts"))
    from visualizations import run_all_visualizations

print("\n" + "=" * 85)
print(f" GENERATING PUBLICATION VISUALIZATION SUITE ({MODEL_NAME}) ".center(85, "="))
print("=" * 85)
run_all_visualizations(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    df_results=df_all,
    loaded_adatas=all_dataset_adata,
    choices=choices,
    dataset_indices=DATASET_INDICES,
)
