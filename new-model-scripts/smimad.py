#!/usr/bin/env python3
"""
smiMAD Multimodal Spatial Clustering Benchmark Script
Faithful reproduction of official smiMAD from source-code/smiMAD-main/
(layers.py, model.py, utils.py, trainer.py, run.py, run_ATAC.py),
wrapped in the multi-dataset, multi-seed 8-metric benchmark suite.
"""

import argparse
import math
import os
import random
import shutil
import sys
import time
import warnings
from typing import Optional

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse
import seaborn as sns
import sklearn
import sklearn.preprocessing
import sklearn.utils.extmath
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
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
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch.backends import cudnn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================================================
# 1. LAYERS (exact from source-code/smiMAD-main/layers.py)
# ================================================================================================

class GraphConvolution(Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 (layers.py:8-40)."""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MultiGCN(nn.Module):
    """Multi-layer GCN branch (layers.py:43-67)."""
    def __init__(self, input_dim, hidden_dim, out_dim, dropout=0.0, num_layers=2):
        super(MultiGCN, self).__init__()
        self.dropout = dropout
        self.conv_layers = nn.ModuleList()

        in_dim = input_dim
        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                out_dim_layer = out_dim
            else:
                out_dim_layer = hidden_dim

            conv_layer = GraphConvolution(in_dim, out_dim_layer)
            self.conv_layers.append(conv_layer)
            in_dim = out_dim_layer

    def forward(self, x, adj):
        for layer_idx, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, adj)
            if layer_idx != len(self.conv_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class self_gating(nn.Module):
    """Self-gating attention fusion module (layers.py:70-85)."""
    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.0, temperature=1.0):
        super(self_gating, self).__init__()
        self.temperature = temperature
        self.gate_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, feature_tensor):
        attention_scores = self.gate_layer(feature_tensor) / self.temperature
        gating_weights = torch.softmax(attention_scores, dim=1)
        weighted_output = (gating_weights * feature_tensor).sum(dim=1)
        return weighted_output, gating_weights


class RNA_Decoder(torch.nn.Module):
    """ZINB RNA decoder (layers.py:87-130)."""
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True),
        })

        output_blocks = []
        for _ in range(3):
            block = torch.nn.Sequential()
            block.add_module("proj", torch.nn.Linear(hidden_dim, out_dim))
            output_blocks.append(block)
        self.pi_head, self.disp_head, self.mean_head = output_blocks

        self.output_activations = torch.nn.ModuleDict({
            "disp": self._create_activation("softplus", 1e-4, 1e4),
            "mean": self._create_activation("exp", 1e-5, 1e6),
        })

    def _create_activation(self, func_type, min_val, max_val):
        class CustomActivation(torch.nn.Module):
            def forward(self, x):
                if func_type == "softplus":
                    output = F.softplus(x)
                elif func_type == "exp":
                    output = torch.exp(x)
                return torch.clamp(output, min_val, max_val)
        return CustomActivation()

    def forward(self, latent):
        features = self.feature_net["linear"](latent)
        features = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](features)

        pi_param = torch.sigmoid(self.pi_head(processed))
        disp_param = self.output_activations["disp"](self.disp_head(processed))
        mean_param = self.output_activations["mean"](self.mean_head(processed))

        return pi_param, disp_param, mean_param


class ATAC_Decoder(torch.nn.Module):
    """Bernoulli ATAC decoder (layers.py:132-158)."""
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True),
        })

        self.prob_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )

        self.output_activations = torch.nn.ModuleDict({
            "prob": torch.nn.Sigmoid()
        })

    def forward(self, latent):
        hidden = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](hidden)
        processed = self.feature_net["activation"](normalized)

        logits = self.prob_head(processed)
        probabilities = self.output_activations["prob"](logits)

        return probabilities


class ADT_Decoder(torch.nn.Module):
    """ADT decoder (layers.py:160-185)."""
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True),
        })

        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )

        self.output_activations = torch.nn.ModuleDict({
            "mean": torch.nn.Identity()
        })

    def forward(self, latent):
        features = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](normalized)

        mean_param = self.output_activations["mean"](self.mean_head(processed))
        return mean_param


class Decoder(torch.nn.Module):
    """Common continuous feature decoder (layers.py:187-212)."""
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.feature_net = torch.nn.ModuleDict({
            "linear": torch.nn.Linear(input_dim, hidden_dim),
            "normalization": torch.nn.BatchNorm1d(hidden_dim),
            "activation": torch.nn.ReLU(inplace=True),
        })

        self.mean_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, out_dim)
        )

        self.output_activations = torch.nn.ModuleDict({
            "mean": torch.nn.Identity()
        })

    def forward(self, latent):
        features = self.feature_net["linear"](latent)
        normalized = self.feature_net["normalization"](features)
        processed = self.feature_net["activation"](normalized)

        mean_param = self.output_activations["mean"](self.mean_head(processed))
        return mean_param


# ================================================================================================
# 2. MODEL ARCHITECTURE (exact from source-code/smiMAD-main/model.py)
# ================================================================================================

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super(Encoder, self).__init__()

        self.spatial_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)
        self.feature_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)
        self.shared_branch = MultiGCN(input_dim, hidden_dim, output_dim, dropout)

        self.gate_mechanism = self_gating(output_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout),
        )

        self.residual_scaling = nn.Parameter(torch.ones(1))

    def compute_embeddings(self, features, graph, specific_branch, shared_branch):
        return {
            "specific": specific_branch(features, graph),
            "shared": shared_branch(features, graph),
        }

    def encode(self, features, spatial_graph, feature_graph):
        spatial_embeddings = self.compute_embeddings(features, spatial_graph, self.spatial_branch, self.shared_branch)
        feature_embeddings = self.compute_embeddings(features, feature_graph, self.feature_branch, self.shared_branch)

        modality_embeddings = torch.stack(
            [
                spatial_embeddings["specific"],
                spatial_embeddings["shared"],
                feature_embeddings["shared"],
                feature_embeddings["specific"],
            ],
            dim=1,
        )

        fused_representation, attention_weights = self.gate_mechanism(modality_embeddings)
        fused_output = self.residual_scaling * self.projection_head(fused_representation)

        return (
            fused_output,
            spatial_embeddings["specific"],
            spatial_embeddings["shared"],
            feature_embeddings["shared"],
            feature_embeddings["specific"],
            attention_weights,
        )

    def forward(self, features, spatial_graph, feature_graph):
        return self.encode(features, spatial_graph, feature_graph)


class smiMAD(nn.Module):
    def __init__(self, omics1_dim, omics2_dim, hidden_dim=64, latent_dim=16, modality_type='adt', common=False):
        super(smiMAD, self).__init__()

        self.omics1_encoder = Encoder(omics1_dim, hidden_dim, latent_dim)
        self.omics2_encoder = Encoder(omics2_dim, hidden_dim, latent_dim)

        self.latent_gating = self_gating(latent_dim)

        if not common:
            self.omics1_decoder = RNA_Decoder(latent_dim, hidden_dim, omics1_dim)
            if modality_type.lower() == 'adt':
                self.omics2_decoder = ADT_Decoder(latent_dim, hidden_dim, omics2_dim)
            else:
                self.omics2_decoder = ATAC_Decoder(latent_dim, hidden_dim, omics2_dim)
        else:
            self.omics1_decoder = Decoder(latent_dim, hidden_dim, omics1_dim)
            self.omics2_decoder = Decoder(latent_dim, hidden_dim, omics2_dim)

        self.modality_type = modality_type
        self.common = common

    def forward(self, omics1_features, omics2_features, spatial_graph, feature_graph_omics1, feature_graph_omics2):
        (
            latent_omics1,
            spatial_specific1,
            spatial_shared1,
            feature_shared1,
            feature_specific1,
            attention_weights1,
        ) = self.omics1_encoder(omics1_features, spatial_graph, feature_graph_omics1)
        (
            latent_omics2,
            spatial_specific2,
            spatial_shared2,
            feature_shared2,
            feature_specific2,
            attention_weights2,
        ) = self.omics2_encoder(omics2_features, spatial_graph, feature_graph_omics2)

        fused_latent, attention_weights_fusion = self.latent_gating(
            torch.stack([latent_omics1, latent_omics2], dim=1)
        )

        if not self.common:
            pi, dispersion, mean = self.omics1_decoder(fused_latent)
            modality2_reconstruction = self.omics2_decoder(fused_latent)

            if self.modality_type == 'atac':
                modality2_reconstruction = torch.sigmoid(modality2_reconstruction)

            return (
                pi,
                dispersion,
                mean,
                modality2_reconstruction,
                spatial_specific1,
                spatial_shared1,
                feature_shared1,
                feature_specific1,
                spatial_specific2,
                spatial_shared2,
                feature_shared2,
                feature_specific2,
                fused_latent,
                attention_weights1,
                attention_weights2,
                attention_weights_fusion,
            )
        else:
            modality1_reconstruction = self.omics1_decoder(fused_latent)
            modality2_reconstruction = self.omics2_decoder(fused_latent)

            return (
                modality1_reconstruction,
                modality2_reconstruction,
                spatial_specific1,
                spatial_shared1,
                feature_shared1,
                feature_specific1,
                spatial_specific2,
                spatial_shared2,
                feature_shared2,
                feature_specific2,
                fused_latent,
                attention_weights1,
                attention_weights2,
                attention_weights_fusion,
            )


# ================================================================================================
# 3. UTILS, PREPROCESSING, GRAPH CONSTRUCTION & LOSSES (exact from source-code/smiMAD-main/utils.py)
# ================================================================================================

def fix_seed(seed=2024):
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


def construct_adjacency_by_spatial(spatial_coords, n_neighbors=3):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    n = spatial_coords.shape[0]
    rows = np.repeat(indices[:, 0], n_neighbors)
    cols = indices[:, 1:].flatten()

    adj = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(n, n)).toarray()
    adj = adj + adj.T
    return np.where(adj > 0, 1, 0)


def spatial_construct_graph(adata, radius=0.2, normalize=True):
    coords = np.asarray(adata.obsm['spatial'], dtype=np.float64)

    if normalize:
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = np.clip(max_vals - min_vals, a_min=np.finfo(float).eps, a_max=None)
        coords = (coords - min_vals) / range_vals

    sparse_graph = radius_neighbors_graph(
        X=coords,
        radius=radius,
        mode='connectivity',
        include_self=False,
        metric='euclidean',
    )
    adjacency = sparse_graph.toarray().astype(np.float32)
    return adjacency


def pca(adata, use_reps=None, n_comps=10):
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        if scipy.sparse.issparse(adata.X):
            feat_pca = pca_model.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca_model.fit_transform(adata.X)
    return feat_pca


def construct_adjacency_by_features(adata, n_neighbors=2, use_pca=True, n_comps=50, Mtype='RNA'):
    if use_pca:
        if Mtype == 'RNA' or Mtype == 'ADT':
            pca_model = PCA(n_components=n_comps)
            if scipy.sparse.issparse(adata.X):
                features = pca_model.fit_transform(adata.X.toarray())
            else:
                features = pca_model.fit_transform(adata.X)
        else:
            features = adata.obsm['X_lsi'].copy()
    else:
        features = adata.obsm['feat'].copy()

    adj = kneighbors_graph(
        features,
        n_neighbors=n_neighbors,
        mode="distance",
        metric="correlation",
        include_self=False,
    ).toarray()

    # Zero-variance cells (e.g. spots with 0 counts in top HVGs) yield NaN correlation distances
    adj = np.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=0.0)
    adj = adj + adj.T
    adata.obsm['adj_feature'] = np.where(adj > 1, 1, adj)
    return adata


def construct_neighbor_graph(adata_omics1, adata_omics2, use_pca=True, adata_omics2_type='ADT',
                             spatial_neighbors=3, omics1_neighbors=20, omics2_neighbors=20,
                             omics1_comps=10, omics2_comps=10, radius=0.2):
    spatial_coords = adata_omics1.obsm['spatial']
    spatial_adj = construct_adjacency_by_spatial(spatial_coords, n_neighbors=spatial_neighbors)
    adata_omics1.obsm['adj_spatial'] = spatial_adj
    adata_omics2.obsm['adj_spatial'] = spatial_adj

    adata_omics1 = construct_adjacency_by_features(
        adata_omics1, n_neighbors=omics1_neighbors, use_pca=use_pca, n_comps=omics1_comps
    )
    adata_omics2 = construct_adjacency_by_features(
        adata_omics2, n_neighbors=omics2_neighbors, use_pca=use_pca,
        n_comps=omics2_comps, Mtype=adata_omics2_type
    )

    adata_omics2.obsm['adjacency'] = spatial_construct_graph(adata_omics1, radius=radius)
    return adata_omics1, adata_omics2


def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell (utils.py:276-291)."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    X_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X_arr)
    return adata


def tfidf(X):
    """TF-IDF normalization (following Seurat v3 approach, utils.py:309-319)."""
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


def lsi(adata: anndata.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    """LSI dimensionality reduction (exact from utils.py:295-306)."""
    X = tfidf(adata.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi_std = X_lsi.std(axis=1, ddof=1, keepdims=True)
    X_lsi_std = np.where(X_lsi_std == 0, 1.0, X_lsi_std)
    X_lsi /= X_lsi_std
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def pre_process_anndata(adata_omics1, adata_omics2, data_type='10x',
                        spatial_neighbors=20, omics1_neighbors=10, omics2_neighbors=10, radius=0.1):
    """Exact preprocessing from source-code/smiMAD-main/utils.py:133-256."""
    adata_omics1 = adata_omics1.copy()
    adata_omics2 = adata_omics2.copy()
    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    if data_type in ('Simulation', 'spatial‐CITE‐seq', 'Stereo-CITE-seq'):
        if data_type == 'Simulation':
            sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
            adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable].copy()

        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)

        adata_omics1, adata_omics2 = construct_neighbor_graph(
            adata_omics1, adata_omics2,
            omics1_comps=adata_omics2.shape[1] - 1,
            omics2_comps=adata_omics2.shape[1] - 1,
            spatial_neighbors=spatial_neighbors,
            omics1_neighbors=omics1_neighbors,
            omics2_neighbors=omics2_neighbors,
            radius=radius,
        )

    elif data_type == 'MISAR-seq':
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)

        adata_omics1 = adata_omics1[:, adata_omics1.var.highly_variable].copy()
        adata_omics2 = adata_omics2[adata_omics1.obs_names, adata_omics2.var.highly_variable].copy()

        lsi(adata_omics2, use_highly_variable=False, n_components=50)

        adata_omics1, adata_omics2 = construct_neighbor_graph(
            adata_omics1, adata_omics2,
            adata_omics2_type='ATAC',
            omics1_comps=50,
            omics2_comps=50,
            spatial_neighbors=spatial_neighbors,
            omics1_neighbors=omics1_neighbors,
            omics2_neighbors=omics2_neighbors,
            radius=radius,
        )

    elif data_type == 'Spatial-epigenome-transcriptome':
        sc.pp.filter_genes(adata_omics1, min_cells=1)
        sc.pp.filter_cells(adata_omics1, min_genes=1)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=50)

        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        if 'X_lsi' not in adata_omics2.obsm:
            sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
            lsi(adata_omics2, use_highly_variable=False, n_components=51)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

        adata_omics1, adata_omics2 = construct_neighbor_graph(
            adata_omics1, adata_omics2,
            use_pca=False,
            spatial_neighbors=spatial_neighbors,
            omics1_neighbors=omics1_neighbors,
            omics2_neighbors=omics2_neighbors,
            radius=radius,
        )

    elif data_type in ('SPOTS', 'DBiT-seq', 'DBiT‑seq'):
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)

        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

        adata_omics1, adata_omics2 = construct_neighbor_graph(
            adata_omics1, adata_omics2,
            use_pca=False,
            spatial_neighbors=spatial_neighbors,
            omics1_neighbors=omics1_neighbors,
            omics2_neighbors=omics2_neighbors,
            radius=radius,
        )

    elif data_type == '10x':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
        adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)

        # Protein / ADT
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)

        adata_omics1, adata_omics2 = construct_neighbor_graph(
            adata_omics1, adata_omics2,
            use_pca=False,
            spatial_neighbors=spatial_neighbors,
            omics1_neighbors=omics1_neighbors,
            omics2_neighbors=omics2_neighbors,
            radius=radius,
        )
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    return adata_omics1, adata_omics2


def clustering(adata, num_cluster=5, used_obsm='fused_latent', modelNames='EEE', random_seed=2024):
    """
    Clustering using the R mclust algorithm (official utils.py:258-274,
    using an explicit R matrix to avoid mclust dimnames failures with newer
    R/mclust combinations.
    """
    import rpy2.robjects as robjects
    from rpy2.robjects.vectors import FloatVector, StrVector

    robjects.r.library("mclust")
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    np.random.seed(random_seed)

    X = np.array(adata.obsm[used_obsm], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{used_obsm} must be a two-dimensional array, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{used_obsm} contains NaN or infinite values")
    if X.shape[0] <= num_cluster:
        raise ValueError(
            f"mclust requires more observations than clusters: "
            f"{X.shape[0]} observations, {num_cluster} clusters"
        )

    # Construct the matrix in R's column-major order and set names before
    # Mclust runs its input validation. This avoids a dimnames assignment
    # failure reported with mclust 6.1.4 and newer R/rpy2 combinations.
    r_data = robjects.r.matrix(
        FloatVector(X.ravel(order="F")),
        nrow=X.shape[0],
        ncol=X.shape[1],
    )
    set_colnames = robjects.r(
        "function(x, names) { colnames(x) <- names; x }"
    )
    r_data = set_colnames(
        r_data,
        StrVector([f"V{i}" for i in range(1, X.shape[1] + 1)]),
    )
    res = rmclust(r_data, num_cluster, modelNames)

    mclust_res = np.array(res.rx2('classification')) if hasattr(res, 'rx2') else np.array(res[-2])

    adata.obs['mclust'] = mclust_res.astype('int')
    return adata


def mmd_gaussian_loss(emb1, emb2, sigma=1.0):
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

    def gaussian_kernel(x, y):
        x_sqnorms = torch.sum(x ** 2, dim=1, keepdim=True)
        y_sqnorms = torch.sum(y ** 2, dim=1, keepdim=True)
        pairwise_dist = x_sqnorms - 2 * torch.mm(x, y.t()) + y_sqnorms.t()
        return torch.exp(-pairwise_dist / (2 * sigma ** 2))

    k_11 = gaussian_kernel(emb1, emb1)
    k_22 = gaussian_kernel(emb2, emb2)
    k_12 = gaussian_kernel(emb1, emb2)
    return k_11.mean() + k_22.mean() - 2 * k_12.mean()


def spatial_consistency_loss(
    fused_latent, adjacency, epoch, total_epochs=5000,
    start_low=0.1, end_low=0.3,
    start_high=0.9, end_high=0.7,
):
    similarity = torch.matmul(fused_latent, fused_latent.T)
    norm = torch.norm(fused_latent, p=2, dim=1).reshape((fused_latent.shape[0], 1))
    similarity = similarity / (norm @ norm.T + 1e-8)
    similarity = similarity - torch.diag_embed(torch.diag(similarity))

    similarity = (similarity + 1) / 2  # map to [0, 1]

    progress = epoch / total_epochs
    margin_low = start_low + (end_low - start_low) * progress
    margin_high = start_high + (end_high - start_high) * progress

    pos_mask = adjacency > 0
    pos_violation = (similarity < margin_high) & pos_mask
    pos_loss = F.relu(margin_high - similarity) * pos_violation.float()
    pos_loss = pos_loss.sum() / (pos_violation.sum() + 1e-8)

    adjacency_neg = 1.0 - adjacency
    adjacency_neg.fill_diagonal_(0)
    neg_mask = adjacency_neg > 0
    neg_violation = (similarity > margin_low) & neg_mask
    neg_loss = F.relu(similarity - margin_low) * neg_violation.float()
    neg_loss = neg_loss.sum() / (neg_violation.sum() + 1e-8)

    return 0.5 * (pos_loss + neg_loss)


def decorrelation_loss(spec1, spec2):
    cos_sim = F.cosine_similarity(spec1, spec2, dim=1)
    return (cos_sim ** 2).mean()


def rna_loss(y_true, y_pred, pi, theta, ridge_lambda=0.0, scale_factor=1.0, mean=True, eps=1e-10):
    theta = torch.minimum(theta, torch.tensor(1e6, device=theta.device, dtype=theta.dtype))
    y_pred = y_pred * scale_factor

    t1 = torch.lgamma(theta + eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + eps))) + y_true * (
        torch.log(theta + eps) - torch.log(y_pred + eps)
    )
    nb_term = t1 + t2

    zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
    nb_case = nb_term - torch.log(1.0 - pi + eps)

    result = torch.where(y_true < 1e-8, zero_case, nb_case)
    result += ridge_lambda * torch.square(pi)
    result = torch.nan_to_num(result, nan=float('inf'))

    return torch.mean(result) if mean else result


def atac_loss(p, target):
    return F.binary_cross_entropy(p, target)


def adt_loss(mean, target):
    return F.huber_loss(mean, target)


# ================================================================================================
# 4. TRAINER (exact from source-code/smiMAD-main/trainer.py)
# ================================================================================================

def train_smimad(adata_omics1, adata_omics2, n_cluster=5, Epoch=5000, lr=0.001,
                 weight_decay=0.1, device='cuda:0', other_type='adt'):
    common = 'feat' in adata_omics1.obsm
    if not common:
        if scipy.sparse.issparse(adata_omics1.X):
            RNA = torch.tensor(adata_omics1.X.toarray()).float().to(device)
        else:
            RNA = torch.tensor(np.asarray(adata_omics1.X)).float().to(device)

        if other_type == 'adt':
            if scipy.sparse.issparse(adata_omics2.X):
                ADT = torch.tensor(adata_omics2.X.toarray()).float().to(device)
            else:
                ADT = torch.tensor(np.asarray(adata_omics2.X)).float().to(device)
        else:
            if scipy.sparse.issparse(adata_omics2.X):
                ATAC = adata_omics2.X.toarray().copy()
            else:
                ATAC = np.asarray(adata_omics2.X).copy()
            ATAC = np.nan_to_num(ATAC, nan=0.0, posinf=0.0, neginf=0.0)
            ATAC[ATAC != 0] = 1
            ATAC = torch.tensor(ATAC).float().to(device)
    else:
        RNA = torch.tensor(np.ascontiguousarray(adata_omics1.obsm['feat'])).float().to(device)
        ADT = torch.tensor(np.ascontiguousarray(adata_omics2.obsm['feat'])).float().to(device)

    spatial_adj = torch.tensor(np.ascontiguousarray(adata_omics1.obsm['adj_spatial'])).float().to(device)
    adj_feature_f1 = torch.tensor(np.ascontiguousarray(adata_omics1.obsm['adj_feature'])).float().to(device)
    adj_feature_f2 = torch.tensor(np.ascontiguousarray(adata_omics2.obsm['adj_feature'])).float().to(device)
    adjacency = torch.tensor(np.ascontiguousarray(adata_omics2.obsm['adjacency'])).float().to(device)

    if not common:
        if other_type == 'adt':
            model = smiMAD(RNA.shape[1], ADT.shape[1]).to(device)
        else:
            model = smiMAD(RNA.shape[1], ATAC.shape[1], modality_type='atac').to(device)
    else:
        model = smiMAD(RNA.shape[1], ADT.shape[1], common=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    progress_bar = tqdm(range(Epoch), desc="Training Progress, loss: 0.0000")
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()

        if not common:
            if other_type == 'adt':
                (pi, dispersion, mean, modality2_reconstruction,
                 spatial_specific1, spatial_shared1, feature_shared1, feature_specific1,
                 spatial_specific2, spatial_shared2, feature_shared2, feature_specific2,
                 fused_latent, _, _, _) = model(RNA, ADT, spatial_adj, adj_feature_f1, adj_feature_f2)
            else:
                (pi, dispersion, mean, modality2_reconstruction,
                 spatial_specific1, spatial_shared1, feature_shared1, feature_specific1,
                 spatial_specific2, spatial_shared2, feature_shared2, feature_specific2,
                 fused_latent, _, _, _) = model(RNA, ATAC, spatial_adj, adj_feature_f1, adj_feature_f2)

            rec_1 = rna_loss(y_true=RNA, y_pred=mean, pi=pi, theta=dispersion,
                             ridge_lambda=0.0, scale_factor=1.0, mean=True)
            if other_type == 'adt':
                rec_2 = adt_loss(modality2_reconstruction, ADT)
            else:
                rec_2 = atac_loss(modality2_reconstruction, ATAC)
        else:
            (modality1_reconstruction, modality2_reconstruction,
             spatial_specific1, spatial_shared1, feature_shared1, feature_specific1,
             spatial_specific2, spatial_shared2, feature_shared2, feature_specific2,
             fused_latent, _, _, _) = model(RNA, ADT, spatial_adj, adj_feature_f1, adj_feature_f2)
            rec_1 = adt_loss(modality1_reconstruction, RNA)
            rec_2 = adt_loss(modality2_reconstruction, ADT)

        mmd1 = mmd_gaussian_loss(spatial_shared1, feature_shared1)
        mmd2 = mmd_gaussian_loss(spatial_shared2, feature_shared2)
        spl = spatial_consistency_loss(fused_latent, adjacency, epoch)
        decl = (decorrelation_loss(spatial_specific1, feature_specific1)
                + decorrelation_loss(spatial_specific2, feature_specific2))

        loss = rec_1 + rec_2 + mmd1 + mmd2 + spl + decl

        loss.backward()
        optimizer.step()

        progress_bar.set_description(
            f"Training Progress, loss: {loss:.4f}, rec1: {rec_1:.4f}, rec2: {rec_2:.4f}, "
            f"mmd1: {mmd1:.4f}, mmd2: {mmd2:.4f}, sp: {spl:.4f}, dec: {decl:.4f}"
        )

    model.eval()
    with torch.no_grad():
        if not common:
            if other_type == 'adt':
                (_, _, _, _, _, _, _, _, _, _, _, _, fused_latent,
                 attention_weights1, attention_weights2, attention_weights_fusion) = model(
                    RNA, ADT, spatial_adj, adj_feature_f1, adj_feature_f2
                )
            else:
                (_, _, _, _, _, _, _, _, _, _, _, _, fused_latent,
                 attention_weights1, attention_weights2, attention_weights_fusion) = model(
                    RNA, ATAC, spatial_adj, adj_feature_f1, adj_feature_f2
                )
        else:
            (_, _, _, _, _, _, _, _, _, _, fused_latent,
             attention_weights1, attention_weights2, attention_weights_fusion) = model(
                RNA, ADT, spatial_adj, adj_feature_f1, adj_feature_f2
            )

    adata_out = adata_omics2.copy()
    adata_out.obsm['fused_latent'] = fused_latent.detach().cpu().numpy()
    adata_out.obsm['smiMAD'] = adata_out.obsm['fused_latent'].copy()
    adata_out.obsm['attention_weights1'] = attention_weights1.detach().cpu().numpy()
    adata_out.obsm['attention_weights2'] = attention_weights2.detach().cpu().numpy()
    adata_out.obsm['attention_weights_fusion'] = attention_weights_fusion.detach().cpu().numpy()
    clustering(adata_out, num_cluster=n_cluster, used_obsm='fused_latent', random_seed=2024)
    adata_out.obs['smiMAD'] = adata_out.obs['mclust'].astype(str).astype('category')
    return adata_out


# ================================================================================================
# 5. ARGUMENT PARSING & BENCHMARK CONFIGURATION (matching new-scripts/ & new-model-scripts/)
# ================================================================================================

parser = argparse.ArgumentParser(description="Run smiMAD multimodal spatial clustering benchmark")
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--epoch', type=int, default=None, help="Override dataset-specific official epoch count")
parser.add_argument('--n_cluster', type=int, default=None, help="Official cluster count; derived from annotation when omitted")
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

MODEL_NAME = "smiMAD"
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
        # Official smiMAD RNA+ADT ('10x') configuration (utils.py:230-251 & README.md:31)
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        smimad_datatype = '10x'
        other_type = 'adt'
        spatial_neighbors = 17
        omics1_neighbors = 20
        omics2_neighbors = 20
        radius = 0.08
        default_epochs = 1000
    else:
        # Official smiMAD MISAR-seq RNA+ATAC configuration (utils.py:158-176 & README.md:35)
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        smimad_datatype = 'MISAR-seq'
        other_type = 'atac'
        spatial_neighbors = 18
        omics1_neighbors = 5
        omics2_neighbors = 5
        radius = 0.05
        default_epochs = 3000

    n_epochs = args.epoch if args.epoch is not None else default_epochs

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
    adata_omics1_raw = sc.read_h5ad(rna_path)
    adata_omics2_raw = sc.read_h5ad(other_path)

    adata_omics1_raw.var_names_make_unique()
    adata_omics2_raw.var_names_make_unique()

    anno_df = pd.read_csv(annotation_path, index_col=0)
    adata_omics1_raw.obs['ground_truth'] = anno_df[gt_column]
    adata_omics2_raw.obs['ground_truth'] = anno_df[gt_column]

    # ------------------ Preprocessing ------------------
    fix_seed(2024)
    adata_omics1, adata_omics2 = pre_process_anndata(
        adata_omics1_raw,
        adata_omics2_raw,
        data_type=smimad_datatype,
        spatial_neighbors=spatial_neighbors,
        omics1_neighbors=omics1_neighbors,
        omics2_neighbors=omics2_neighbors,
        radius=radius,
    )

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = int(adata_omics2.obs["ground_truth"].dropna().nunique())
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = args.n_cluster if args.n_cluster is not None else n_ground_truth
    if args.n_cluster is not None:
        print(f"Using explicit official cluster count: {n_cluster}")

    best_ari = -1.0
    best_seed = None
    best_adata = None

    for seed_val in SEEDS:
        print("\n" + "-" * 60)
        print(f" Dataset: {dataset_name} | Seed: {seed_val} ".center(60, "-"))
        print("-" * 60)

        seed_start_time = time.time()
        fix_seed(seed_val)

        # 1. Model Training + Mclust Clustering
        adata = train_smimad(
            adata_omics1=adata_omics1,
            adata_omics2=adata_omics2,
            n_cluster=n_cluster,
            Epoch=n_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            other_type=other_type,
        )

        duration_sec = round(time.time() - seed_start_time, 2)

        # 2. Score Generating (8 metrics + Duration, masked by valid_mask)
        if 'anno' not in adata.obs.columns and 'ground_truth' in adata.obs.columns:
            adata.obs['anno'] = adata.obs['ground_truth']

        valid_mask = adata.obs['ground_truth'].notna()
        y_true = adata.obs.loc[valid_mask, 'ground_truth'].astype(str)
        y_pred_eval = adata.obs.loc[valid_mask, 'smiMAD'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred_eval)
        nmi = normalized_mutual_info_score(y_true, y_pred_eval)
        ami = adjusted_mutual_info_score(y_true, y_pred_eval)
        homogeneity = homogeneity_score(y_true, y_pred_eval)
        v_measure = v_measure_score(y_true, y_pred_eval)

        joint_feat = adata.obsm['fused_latent'][valid_mask.values]
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

        print(f"\nResult for dataset: {dataset_name} | seed: {seed_val}")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil_score:.4f} | CHI: {chi_score:.2f} | DBI: {dbi_score:.4f} | Duration: {duration_sec:.2f}s")

        res_dict = {
            'dataset': dataset_name,
            'seed': seed_val,
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
            best_seed = seed_val
            best_adata = adata.copy()
            best_adata.obsm[f'{MODEL_NAME}_emb'] = best_adata.obsm['fused_latent'].copy()
            best_adata.obs[f'{MODEL_NAME}_domain'] = best_adata.obs['smiMAD'].copy()
            best_adata.obs['domain'] = best_adata.obs['smiMAD'].copy()
            best_adata.uns['selected_seed'] = best_seed

        # ------------------ Plotting & Visualization ------------------
        fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
        sc.pp.neighbors(adata, use_rep='fused_latent', n_neighbors=10)
        sc.tl.umap(adata)

        sc.pl.umap(adata, color='smiMAD', ax=ax_list[0], title='smiMAD', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='smiMAD', ax=ax_list[1], title='smiMAD', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='anno', ax=ax_list[2], title='anno', s=20, show=False)

        plt.tight_layout(w_pad=0.3)
        os.makedirs("results", exist_ok=True)
        plot_filename = os.path.join(VISUALS_DIR, f"{dataset_name}_seed_{seed_val}.png")
        plt.savefig(plot_filename, dpi=150)
        plt.savefig(f"results/{MODEL_NAME}_{dataset_name}_seed_{seed_val}.png", dpi=150)
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
# 6. GLOBAL SUMMARY & PUBLICATION VISUALIZATION SUITE
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
