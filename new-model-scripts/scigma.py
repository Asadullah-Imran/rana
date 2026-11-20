#!/usr/bin/env python3
"""
SCIGMA Multimodal Spatial Clustering Benchmark Script
Faithful reproduction of official SCIGMA from source-code/SCIGMA/SCIGMA-main/
and source-code/SCIGMA/SCIGMA_analyses-main/scripts/ (SCIGMA_SPOTS.py & SCIGMA_ATAC.py),
wrapped in the multi-dataset, multi-seed 8-metric benchmark suite.
"""

import argparse
import enum
import gc
import os
import random
import shutil
import sys
import time
import warnings

import anndata
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import seaborn as sns
import sklearn
import sklearn.preprocessing
import sklearn.utils.extmath
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
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
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ================================================================================================
# 1. UTILS (exact from source-code/SCIGMA/SCIGMA-main/utils.py)
# ================================================================================================

def seed(seed_val=1000):
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def tfidf(X):
    """TF-IDF normalization (following the Seurat v3 approach, utils.py:56-66)."""
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


def lsi(adata: anndata.AnnData, n_components=20, use_highly_variable=True, **kwargs):
    """LSI analysis (following the Seurat v3 approach, utils.py:37-53)."""
    if use_highly_variable:
        sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000)
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi_std = X_lsi.std(axis=1, ddof=1, keepdims=True)
    X_lsi_std = np.where(X_lsi_std == 0, 1.0, X_lsi_std)
    X_lsi /= X_lsi_std
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=1000):
    """
    Clustering using the mclust algorithm (official utils.py:69-89,
    using rpy2 localconverter for compatibility with Python 3.11+ / rpy2 >= 3.5).
    """
    import rpy2.robjects as robjects
    from rpy2.robjects import default_converter, pandas2ri
    from rpy2.robjects.conversion import localconverter

    np.random.seed(random_seed)
    robjects.r.library("mclust")

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    X = np.array(adata.obsm[used_obsm], dtype=np.float64)
    df = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])

    with localconverter(default_converter + pandas2ri.converter):
        res = rmclust(df, G=num_cluster, modelNames=modelNames)

    if hasattr(res, 'rx2'):
        mclust_res = np.array(res.rx2('classification'))
    elif hasattr(res, 'getbyname'):
        mclust_res = np.array(res.getbyname('classification'))
    else:
        mclust_res = np.array(res['classification'])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('str')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    """Searching corresponding resolution according to given cluster number (utils.py:295-340)."""
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        if count_unique == n_clusters:
            label = 1
            break

    assert label == 1, "Resolution is not found. Please try bigger range or smaller step!."
    return res


def clustering(adata, n_clusters=7, key='emb', add_key='SCIGMA', method='mclust',
               start=0.1, end=3.0, increment=0.01, use_pca=True, n_components=20,
               weight=None, random_seed=1000):
    """Spatial clustering based on the latent representation (utils.py:238-292)."""
    if use_pca:
        pca = PCA(n_components=n_components, random_state=42)
        adata.obsm[key + '_pca'] = pca.fit_transform(adata.obsm[key].copy())
        if weight:
            adata.obsm[key + '_pca'] = adata.obsm[key + '_pca'] * weight

    if method == 'mclust':
        if use_pca:
            adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters, random_seed=random_seed)
        else:
            adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters, random_seed=random_seed)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        if use_pca:
            res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
        else:
            res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


# ================================================================================================
# 2. DATA PREPROCESSING & GRAPH CONSTRUCTION (exact from source-code/SCIGMA/SCIGMA-main/data.py)
# ================================================================================================

def clr_normalize_each_cell(adata, inplace=True):
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()

    X_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X_arr)
    return adata


def feature_graph(adata_omics1, adata_omics2, k=6, mode="connectivity", metric="correlation", include_self=False):
    print("Feature graph 1")
    feature_graph_omics1 = kneighbors_graph(
        adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self, n_jobs=-1
    )
    print("Feature graph 1 finished")
    gc.collect()
    print("Feature graph 2")
    feature_graph_omics2 = kneighbors_graph(
        adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self, n_jobs=-1
    )
    print("Feature graph 2 finished")
    gc.collect()
    return feature_graph_omics1, feature_graph_omics2


def build_network(cell_position, n_neighbors=6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def construct_graph(adjacent):
    n_spot = int(adjacent['x'].max()) + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def construct_combined_graphs(adata1, adata2):
    adj_spatial_omics1 = construct_graph(adata1.uns['adj_spatial'])
    adj_spatial_omics2 = construct_graph(adata2.uns['adj_spatial'])
    adj_feature_omics1 = adata1.obsm['adj_feature']
    adj_feature_omics2 = adata2.obsm['adj_feature']

    # combined graph
    combined_graph_1 = (adj_spatial_omics1 + adj_feature_omics1).sign()
    adj_spatial_omics1 = None
    adj_feature_omics1 = None
    gc.collect()

    combined_graph_2 = (adj_spatial_omics2 + adj_feature_omics2).sign()
    adj_spatial_omics2 = None
    adj_feature_omics2 = None
    gc.collect()

    # normalize the combined graph
    combined_graph_1 = combined_graph_1 + combined_graph_1.transpose()
    indices = combined_graph_1.data > 1
    combined_graph_1.data[indices] = 1
    combined_graph_1 = normalize_adj(combined_graph_1)
    combined_graph_1_sp = nx.from_scipy_sparse_array(
        sparse.coo_matrix(
            (combined_graph_1.coalesce().values().numpy(), combined_graph_1.coalesce().indices().numpy()),
            shape=combined_graph_1.shape,
        )
    )
    gc.collect()

    combined_graph_2 = combined_graph_2 + combined_graph_2.transpose()
    indices = combined_graph_2.data > 1
    combined_graph_2.data[indices] = 1
    combined_graph_2 = normalize_adj(combined_graph_2)
    combined_graph_2_sp = nx.from_scipy_sparse_array(
        sparse.coo_matrix(
            (combined_graph_2.coalesce().values().numpy(), combined_graph_2.coalesce().indices().numpy()),
            shape=combined_graph_2.shape,
        )
    )
    gc.collect()

    return {
        'final_combined_graph_1': combined_graph_1,
        'final_combined_graph_2': combined_graph_2,
        'combined_graph_1_sp': combined_graph_1_sp,
        'combined_graph_2_sp': combined_graph_2_sp,
    }


def preprocessing(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=8, feat_neighbors=8):
    """Exact preprocessing from source-code/SCIGMA/SCIGMA-main/data.py:199-417."""
    if datatype not in ['SPOTS', 'Stereo-CITE-seq', 'Spatial-ATAC-RNA-seq']:
        raise ValueError(f"The datatype '{datatype}' is not supported.")

    print("Preprocessing anndatas")
    if datatype == 'SPOTS':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e6)
        sc.pp.log1p(adata_omics1)
        adata_omics1.layers['logcounts'] = adata_omics1.X.copy()
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']].copy()
        sc.pp.pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']

        # Protein / ADT
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        adata_omics2.layers['logcounts'] = adata_omics2.X.copy()
        sc.pp.scale(adata_omics2)
        sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()

    elif datatype == 'Stereo-CITE-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=80)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e6)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']].copy()
        sc.pp.pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
        adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']

        # Protein / ADT
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        sc.pp.pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_pca'].copy()

    elif datatype == 'Spatial-ATAC-RNA-seq':
        # RNA
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        sc.pp.filter_genes(adata_omics2, min_cells=10)
        sc.pp.filter_cells(adata_omics1, min_genes=200)

        adata_omics1.layers['counts'] = adata_omics1.X.copy()
        sc.pp.normalize_total(adata_omics1, target_sum=1e6)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)
        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000, layer="counts")

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']].copy()
        sc.pp.pca(adata_omics1_high, n_comps=50)
        adata_omics1.obsm['feat'] = adata_omics1_high.obsm['X_pca']

        # ATAC
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
        if 'X_lsi' not in adata_omics2.obsm:
            lsi(adata_omics2, n_components=50)
        adata_omics2.obsm['feat'] = adata_omics2.obsm['X_lsi'].copy()

    print("Constructing spatial graphs")
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = build_network(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    gc.collect()

    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = build_network(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    gc.collect()

    print("Constructing feature graphs")
    feature_graph_omics1, feature_graph_omics2 = feature_graph(adata_omics1, adata_omics2, k=feat_neighbors)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    gc.collect()

    return {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}


# ================================================================================================
# 3. GRAPH SAMPLING (exact from source-code/SCIGMA/SCIGMA-main/graph_sampling.py)
# ================================================================================================

class ForestFire:
    def __init__(self):
        self.G1 = nx.Graph()

    def forestfire(self, G, size):
        list_nodes = list(G.nodes())
        dictt = set()
        random_node = random.sample(list_nodes, 1)[0]
        q = set()
        q.add(random_node)
        while len(self.G1.nodes()) < size:
            if len(q) > 0:
                initial_node = q.pop()
                if initial_node not in dictt:
                    dictt.add(initial_node)
                    neighbours = list(G.neighbors(initial_node))
                    n_burn = random.randint(1, len(neighbours))
                    for x in neighbours[:n_burn]:
                        if len(self.G1.nodes()) < size:
                            self.G1.add_edge(initial_node, x)
                            q.add(x)
                        else:
                            break
                else:
                    continue
            else:
                remaining = list(set(list_nodes) - dictt)
                if not remaining:
                    break
                random_node = random.sample(remaining, 1)[0]
                q.add(random_node)
        q.clear()
        return self.G1


# ================================================================================================
# 4. LOSSES (exact from source-code/SCIGMA/SCIGMA-main/losses.py)
# ================================================================================================

class CrossCLR_onlyIntraModality(nn.Module):
    """CrossCLR Loss between 2 groups of embeddings (exact from losses.py:44-130)."""

    def __init__(self, temperature=0.03, temperature_scale=50, negative_weight=0.8,
                 logger=None, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.temperature = temperature
        self.temperature_scale = temperature_scale
        self.logger = logger
        self.negative_w = negative_weight
        self.device = device

    def compute_loss(self, logits, mask):
        return -torch.log((F.softmax(logits, dim=1) * mask).sum(1))

    def _get_positive_mask(self, batch_size):
        diag = np.eye(batch_size)
        mask = torch.from_numpy(diag)
        mask = (1 - mask).to(self.device)
        return mask

    def forward(self, video_features, text_features, tau, weights=None):
        batch_size = video_features.shape[0]
        self.temperature = torch.sigmoid(tau) / 100

        # Normalize features
        video_features = nn.functional.normalize(video_features, dim=1)
        text_features = nn.functional.normalize(text_features, dim=1)

        # Inter-modality alignment
        logits_per_vid = video_features @ text_features.t()
        logits_per_text = text_features @ video_features.t()

        # Intra-modality alignment
        logits_clstr_vid = video_features @ video_features.t()
        logits_clstr_txt = text_features @ text_features.t()

        logits_per_vid = logits_per_vid / self.temperature
        logits_per_text = logits_per_text / self.temperature
        logits_clstr_vid = logits_clstr_vid / self.temperature
        logits_clstr_txt = logits_clstr_txt / self.temperature

        positive_mask = self._get_positive_mask(video_features.shape[0])
        negatives_vid = logits_clstr_vid * positive_mask
        negatives_txt = logits_clstr_txt * positive_mask

        vid_logits = torch.cat([logits_per_vid, self.negative_w * negatives_vid], dim=1)
        txt_logits = torch.cat([logits_per_text, self.negative_w * negatives_txt], dim=1)

        diag = np.eye(batch_size)
        mask_vid = torch.from_numpy(diag).to(self.device)
        mask_txt = torch.from_numpy(diag).to(self.device)

        mask_neg_v = torch.zeros_like(negatives_vid)
        mask_neg_t = torch.zeros_like(negatives_txt)
        mask_v = torch.cat([mask_vid, mask_neg_v], dim=1)
        mask_t = torch.cat([mask_txt, mask_neg_t], dim=1)

        if weights is not None:
            loss_i = self.compute_loss(vid_logits, mask_v) * weights
            loss_t = self.compute_loss(txt_logits, mask_t) * weights
        else:
            loss_i = self.compute_loss(vid_logits, mask_v)
            loss_t = self.compute_loss(txt_logits, mask_t)

        return (loss_i.mean() + loss_t.mean()) / 2


# ================================================================================================
# 5. MODEL ARCHITECTURE (exact from source-code/SCIGMA/SCIGMA-main/models.py)
# ================================================================================================

class Model(Module):
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2,
                 uncertainty_size, dropout=0.1, layer='GAT',
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(Model, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.uncertainty_size = uncertainty_size
        self.dropout = dropout
        self.layer = layer
        self.device = device

        if self.layer == 'GCN':
            self.encoder_omics1 = GCN(self.dim_in_feat_omics1, self.dim_out_feat_omics1, self.dim_out_feat_omics1, dropout)
            self.decoder_omics1 = GCN(self.dim_out_feat_omics1, self.dim_out_feat_omics1, self.dim_in_feat_omics1, dropout)

            self.encoder_omics2 = GCN(self.dim_in_feat_omics2, self.dim_out_feat_omics2, self.dim_out_feat_omics2, dropout)
            self.decoder_omics2 = GCN(self.dim_out_feat_omics2, self.dim_out_feat_omics2, self.dim_in_feat_omics2, dropout)
        elif self.layer == 'GAT':
            self.encoder_omics1 = GAT(2, [8, 2], [self.dim_in_feat_omics1, self.dim_out_feat_omics1, self.dim_out_feat_omics1])
            self.decoder_omics1 = GAT(2, [2, 8], [self.dim_out_feat_omics1, self.dim_out_feat_omics1, self.dim_in_feat_omics1])

            self.encoder_omics2 = GAT(2, [8, 2], [self.dim_in_feat_omics2, self.dim_out_feat_omics2, self.dim_out_feat_omics2])
            self.decoder_omics2 = GAT(2, [2, 8], [self.dim_out_feat_omics2, self.dim_out_feat_omics2, self.dim_in_feat_omics2])
        elif self.layer == 'GCAT':
            self.encoder_omics1 = GCAT_Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1, self.dim_out_feat_omics1, dropout)
            self.decoder_omics1 = GCAT_Decoder(self.dim_out_feat_omics1, self.dim_out_feat_omics1, self.dim_in_feat_omics1, dropout)

            self.encoder_omics2 = GCAT_Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2, self.dim_out_feat_omics2, dropout)
            self.decoder_omics2 = GCAT_Decoder(self.dim_out_feat_omics2, self.dim_out_feat_omics2, self.dim_in_feat_omics2, dropout)

        self.mod_temp = nn.Sequential(
            nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1 // 2),
            nn.LeakyReLU(),
            nn.Linear(self.dim_out_feat_omics1 // 2, 1),
        )

        self.graph_cross = Attention(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.graph_uncross = nn.Linear(self.dim_out_feat_omics1, self.dim_out_feat_omics1 + self.dim_out_feat_omics2)

    def get_temp(self, emb_combined):
        return self.mod_temp(emb_combined)

    def forward(self, features_omics1, features_omics2, edges_omics1, edges_omics2):
        if self.layer == 'GAT':
            edges_omics1 = edges_omics1.indices().to(self.device)
            edges_omics2 = edges_omics2.indices().to(self.device)

            emb_latent_omics1 = self.encoder_omics1([features_omics1, edges_omics1])[0]
            emb_latent_omics2 = self.encoder_omics2([features_omics2, edges_omics2])[0]

            # between-modality attention aggregation
            emb_latent_combined, attention = self.graph_cross(emb_latent_omics1, emb_latent_omics2)

            emb_latent_recon = self.graph_uncross(emb_latent_combined)
            (emb_latent_omics1_recon, emb_latent_omics2_recon) = torch.split(
                emb_latent_recon, emb_latent_combined.shape[1], dim=1
            )

            emb_recon_omics1 = self.decoder_omics1([emb_latent_omics1_recon, edges_omics1])[0]
            emb_recon_omics2 = self.decoder_omics2([emb_latent_omics2_recon, edges_omics2])[0]

            edges_omics1 = None
            edges_omics2 = None

        elif self.layer == 'GCN':
            adj_omics1 = edges_omics1.to_sparse().to(self.device)
            adj_omics2 = edges_omics2.to_sparse().to(self.device)

            emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_omics1)
            emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_omics2)

            emb_latent_combined, attention = self.graph_cross(emb_latent_omics1, emb_latent_omics2)

            emb_latent_recon = self.graph_uncross(emb_latent_combined)
            (emb_latent_omics1_recon, emb_latent_omics2_recon) = torch.split(
                emb_latent_recon, emb_latent_combined.shape[1], dim=1
            )

            emb_recon_omics1 = self.decoder_omics1(F.leaky_relu(emb_latent_omics1_recon), adj_omics1)
            emb_recon_omics2 = self.decoder_omics2(F.leaky_relu(emb_latent_omics2_recon), adj_omics2)

            adj_omics1 = None
            adj_omics2 = None

        elif self.layer == 'GCAT':
            adj_omics1 = edges_omics1.to_sparse().to(self.device)
            adj_omics2 = edges_omics2.to_sparse().to(self.device)
            edges_omics1 = edges_omics1.indices().to(self.device)
            edges_omics2 = edges_omics2.indices().to(self.device)

            emb_latent_omics1 = self.encoder_omics1(features_omics1, adj_omics1, edges_omics1)
            emb_latent_omics2 = self.encoder_omics2(features_omics2, adj_omics2, edges_omics2)

            emb_latent_combined, attention = self.graph_cross(emb_latent_omics1, emb_latent_omics2)

            emb_latent_recon = self.graph_uncross(emb_latent_combined)
            (emb_latent_omics1_recon, emb_latent_omics2_recon) = torch.split(
                emb_latent_recon, emb_latent_combined.shape[1], dim=1
            )

            emb_recon_omics1 = self.decoder_omics1(emb_latent_omics1_recon, adj_omics1, edges_omics1)
            emb_recon_omics2 = self.decoder_omics2(emb_latent_omics2_recon, adj_omics2, edges_omics2)

            adj_omics1 = None
            adj_omics2 = None
            edges_omics1 = None
            edges_omics2 = None

        return {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'attention_weights': attention,
        }


class Attention(Module):
    """Bahdanau Attention Mechanism (models.py:148-176)."""
    def __init__(self, in_feat, out_feat):
        super(Attention, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat

        self.w = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.v = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, emb1, emb2):
        emb = [
            torch.unsqueeze(torch.squeeze(emb1), dim=1),
            torch.unsqueeze(torch.squeeze(emb2), dim=1),
        ]
        self.emb = torch.cat(emb, dim=1)
        self.score = torch.matmul(torch.tanh(torch.matmul(self.emb, self.w)), self.v)
        self.alpha = F.softmax(torch.squeeze(self.score) + 1e-6, dim=1)
        context = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(context), self.alpha


class GCN_Layer(Module):
    """Modality-specific GNN encoder (models.py:178-197)."""
    def __init__(self, in_feat, out_feat):
        super(GCN_Layer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCN, self).__init__()
        self.gc1 = GCN_Layer(nfeat, nhid)
        self.gc2 = GCN_Layer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class GCAT_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCAT_Encoder, self).__init__()
        self.gc1 = GAT(1, [4], [nfeat, nhid])
        self.gc2 = GCN_Layer(nhid, nhid)
        self.gc3 = GAT(1, [2], [nhid, nclass])
        self.gc4 = GCN_Layer(nclass, nclass)
        self.dropout = dropout

    def forward(self, x, adj, edge):
        x = F.leaky_relu(self.gc1([x, edge])[0])
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc3([x, edge])[0])
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4(x, adj)
        return x


class GCAT_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2):
        super(GCAT_Decoder, self).__init__()
        self.gc1 = GCN_Layer(nfeat, nfeat)
        self.gc2 = GAT(1, [2], [nfeat, nhid])
        self.gc3 = GCN_Layer(nhid, nhid)
        self.gc4 = GAT(1, [4], [nhid, nclass])
        self.dropout = dropout

    def forward(self, x, adj, edge):
        x = F.leaky_relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc2([x, edge])[0])
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.leaky_relu(self.gc3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc4([x, edge])[0]
        return x


class LayerType(enum.Enum):
    IMP1 = 0,
    IMP2 = 1,
    IMP3 = 2


class GAT(torch.nn.Module):
    """Exact GAT implementation from models.py:268-310."""

    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer,
                 add_skip_connection=False, bias=True, dropout=0.1,
                 layer_type=LayerType.IMP3, log_attention_weights=False):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, 'Enter valid arch params.'

        GATLayerCls = get_layer_type(layer_type)
        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayerCls(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i],
                num_out_features=num_features_per_layer[i + 1],
                num_of_heads=num_heads_per_layer[i + 1],
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                bias=bias,
                log_attention_weights=log_attention_weights,
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(*gat_layers)

    def forward(self, data):
        return self.gat_net(data)


class GATLayer(torch.nn.Module):
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, layer_type,
                 concat=True, activation=nn.ELU(), dropout_prob=0.2,
                 add_skip_connection=True, bias=True, log_attention_weights=False):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection

        if layer_type == LayerType.IMP1:
            self.proj_param = nn.Parameter(torch.Tensor(num_of_heads, num_in_features, num_out_features))
        else:
            self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if layer_type == LayerType.IMP1:
            self.scoring_fn_target = nn.Parameter(self.scoring_fn_target.reshape(num_of_heads, num_out_features, 1))
            self.scoring_fn_source = nn.Parameter(self.scoring_fn_source.reshape(num_of_heads, num_out_features, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None

        self.init_params(layer_type)

    def init_params(self, layer_type):
        nn.init.xavier_uniform_(self.proj_param if layer_type == LayerType.IMP1 else self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:
            self.attention_weights = attention_coefficients
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()
        if self.add_skip_connection:
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        if self.concat:
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)
        if self.bias is not None:
            out_nodes_features += self.bias
        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)


class GATLayerImp3(GATLayer):
    src_nodes_dim = 0
    trg_nodes_dim = 1
    nodes_dim = 0
    head_dim = 1

    def __init__(self, num_in_features, num_out_features, num_of_heads, concat=True,
                 activation=nn.ELU(), dropout_prob=0.2, add_skip_connection=True,
                 bias=True, log_attention_weights=False):
        super().__init__(
            num_in_features, num_out_features, num_of_heads, LayerType.IMP3,
            concat, activation, dropout_prob, add_skip_connection, bias, log_attention_weights,
        )

    def forward(self, data):
        in_nodes_features, edge_index = data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        in_nodes_features = self.dropout(in_nodes_features)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        nodes_features_proj = self.dropout(nodes_features_proj)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(
            scores_source, scores_target, nodes_features_proj, edge_index
        )
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)
        attentions_per_edge = self.neighborhood_aware_softmax(
            scores_per_edge, edge_index[self.trg_nodes_dim], num_of_nodes
        )
        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge
        out_nodes_features = self.aggregate_neighbors(
            nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes
        )
        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        return (out_nodes_features, edge_index)

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(
            exp_scores_per_edge, trg_index, num_of_nodes
        )
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)
        size = list(exp_scores_per_edge.shape)
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)
        size[self.nodes_dim] = num_of_nodes
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        trg_index_broadcasted = self.explicit_broadcast(
            edge_index[self.trg_nodes_dim], nodes_features_proj_lifted_weighted
        )
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)
        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)
        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)
        return this.expand_as(other)


def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), f'Expected {LayerType} got {type(layer_type)}.'
    if layer_type == LayerType.IMP3:
        return GATLayerImp3
    raise Exception(f'Layer type {layer_type} not yet supported.')


# ================================================================================================
# 6. SCIGMA TRAINER (exact from source-code/SCIGMA/SCIGMA-main/SCIGMA.py)
# ================================================================================================

def connectivity(data, gamma):
    """Calculates connectivity value for each sample (SCIGMA.py:12-28)."""
    mask = 1 - np.eye(data.shape[0])
    prox_vid = np.mean((data @ data.T) * mask, axis=1)
    max_val = np.max(prox_vid)
    if max_val == 0:
        return list(range(data.shape[0]))
    scores_v = prox_vid / max_val
    return [i for i, x in enumerate(scores_v) if x < gamma]


class SCIGMA:
    def __init__(
        self,
        data,
        seed_num,
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        inference_device=torch.device('cpu'),
        learning_rate=1e-4,
        weight_decay=1e-2,
        clr_weight=0.3,
        contrastive_weight1=0.1,
        contrastive_weight2=0.1,
        recon_weight1=1,
        recon_weight2=1,
        connectivities=None,
        epochs=1000,
        dim_output=64,
        batch_size=6000,
        downweight=100,
        architecture='GAT',
    ):
        seed(seed_num)
        self.data = data
        self.device = device
        self.inference_device = inference_device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_output = dim_output
        self.contrastive_weight1 = contrastive_weight1
        self.contrastive_weight2 = contrastive_weight2
        self.recon_weight1 = recon_weight1
        self.recon_weight2 = recon_weight2
        self.batch_size = batch_size
        self.downweight = downweight

        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']

        if not connectivities:
            connectivities = [i for i in range(len(self.adata_omics1.obsm['feat']))]
        self.connectivities = set(connectivities)
        self.features_omics1 = torch.FloatTensor(np.ascontiguousarray(self.adata_omics1.obsm['feat'].copy(), dtype=np.float32)).to(self.device)
        self.features_omics2 = torch.FloatTensor(np.ascontiguousarray(self.adata_omics2.obsm['feat'].copy(), dtype=np.float32)).to(self.device)
        print(f"Num samples pruned: {len(self.adata_omics1.obsm['feat']) - len(connectivities)}")

        gc.collect()
        print("Creating adjacency matrices")
        adj_data = construct_combined_graphs(self.adata_omics1, self.adata_omics2)
        self.combined_graph_1 = adj_data['final_combined_graph_1']
        self.combined_graph_2 = adj_data['final_combined_graph_2']
        self.combined_graph_1_sp = adj_data['combined_graph_1_sp']
        self.combined_graph_2_sp = adj_data['combined_graph_2_sp']

        self.n_cell_omics1 = self.adata_omics1.n_obs
        self.n_cell_omics2 = self.adata_omics2.n_obs

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

        self.criterion = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight, device=self.device)
        self.criterion2 = CrossCLR_onlyIntraModality(temperature=0.03, negative_weight=clr_weight, device=self.device)

        self.model = Model(
            self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2,
            uncertainty_size=self.features_omics1.shape[0],
            layer=architecture,
            device=self.device,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )

        print("Model ready for training!")

    def train(self, epochs):
        sampler = ForestFire()
        for epoch in tqdm(range(epochs)):
            if epoch % 2 == 0:
                sample = sampler.forestfire(self.combined_graph_1_sp, self.batch_size)
                indices = torch.tensor(list(set([n for n in sample]).intersection(self.connectivities)), dtype=torch.long)
                edge1 = torch.index_select(torch.index_select(self.combined_graph_1, 0, indices), 1, indices).coalesce()
                feat1 = self.features_omics1[indices.to(self.device)]
                edge2 = torch.index_select(torch.index_select(self.combined_graph_2, 0, indices), 1, indices).coalesce()
                feat2 = self.features_omics2[indices.to(self.device)]
            else:
                sample = sampler.forestfire(self.combined_graph_2_sp, self.batch_size)
                indices = torch.tensor(list(set([n for n in sample]).intersection(self.connectivities)), dtype=torch.long)
                edge2 = torch.index_select(torch.index_select(self.combined_graph_2, 0, indices), 1, indices).coalesce()
                feat2 = self.features_omics2[indices.to(self.device)]
                edge1 = torch.index_select(torch.index_select(self.combined_graph_1, 0, indices), 1, indices).coalesce()
                feat1 = self.features_omics1[indices.to(self.device)]
            sample = None

            self.model.train()

            results = self.model(feat1, feat2, edge1, edge2)
            invtau = self.model.get_temp(results['emb_latent_combined'])
            tau = torch.exp(invtau)

            # reconstruction loss
            loss_recon_omics1 = self.recon_weight1 * F.mse_loss(results['emb_recon_omics1'], feat1)
            loss_recon_omics2 = self.recon_weight2 * F.mse_loss(results['emb_recon_omics2'], feat2)

            # contrastive loss
            loss_contrast_mod1 = self.criterion(results['emb_latent_combined'], results['emb_latent_omics1'], tau=tau)
            loss_contrast_mod2 = self.criterion2(results['emb_latent_combined'], results['emb_latent_omics2'], tau=tau)

            # full loss function
            loss = (
                loss_recon_omics1
                + loss_recon_omics2
                + self.contrastive_weight1 * loss_contrast_mod1
                + self.contrastive_weight2 * loss_contrast_mod2
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            feat1 = None
            feat2 = None
            edge1 = None
            edge2 = None

        print("Model training finished!\n")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.eval()
        self.optimizer.zero_grad()
        self.model.device = self.inference_device
        gc.collect()
        self.model = self.model.to(self.inference_device)
        for param in self.model.parameters():
            param.requires_grad = False
        with torch.no_grad():
            results = self.model(
                self.features_omics1.detach().cpu(),
                self.features_omics2.detach().cpu(),
                self.combined_graph_1.coalesce().detach().cpu(),
                self.combined_graph_2.coalesce().detach().cpu(),
            )
            invtau = self.model.get_temp(results['emb_latent_combined'])
            tau = torch.exp(invtau)
        gc.collect()

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)
        emb_recon_omics1 = results['emb_recon_omics1']
        emb_recon_omics2 = results['emb_recon_omics2']

        return {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'emb_latent_combined': emb_combined.detach().cpu().numpy(),
            'emb_recon_omics1': emb_recon_omics1.detach().cpu().numpy(),
            'emb_recon_omics2': emb_recon_omics2.detach().cpu().numpy(),
            'tau': tau.detach().cpu().numpy(),
            'invtau': invtau.detach().cpu().numpy(),
            'attention': results['attention_weights'].detach().cpu().numpy(),
        }


# ================================================================================================
# 7. ARGUMENT PARSING & BENCHMARK CONFIGURATION (matching new-scripts/)
# ================================================================================================

parser = argparse.ArgumentParser(description="Run SCIGMA multimodal spatial clustering benchmark")
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--architecture', type=str, default='GAT', choices=['GAT', 'GCN', 'GCAT'])
parser.add_argument('--tool', type=str, default='mclust', choices=['mclust', 'leiden', 'louvain'])
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

MODEL_NAME = "SCIGMA"
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
        # Official SCIGMA_SPOTS.py RNA+Protein configuration
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        scigma_datatype = 'SPOTS'
        n_neighbors = 6
        feat_neighbors = 6
        dim_output = 20
        batch_size = 2000
        contrastive_weight1 = 1
        contrastive_weight2 = 1
        epochs = 300
        use_connectivity_pruning = True
    else:
        # Official SCIGMA_ATAC.py & SCIGMA_Tutorial.ipynb RNA+ATAC configuration
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        scigma_datatype = 'Spatial-ATAC-RNA-seq'
        n_neighbors = 18
        feat_neighbors = 6
        dim_output = 40
        batch_size = 8000
        contrastive_weight1 = 1e-2
        contrastive_weight2 = 1e-2
        epochs = 600
        use_connectivity_pruning = False

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
    seed(1000)
    data_dict = preprocessing(
        adata_omics1,
        adata_omics2,
        datatype=scigma_datatype,
        n_neighbors=n_neighbors,
        feat_neighbors=feat_neighbors,
    )

    if use_connectivity_pruning:
        X1_pos = data_dict['adata_omics1'].X
        X1_pos = X1_pos.toarray() if scipy.sparse.issparse(X1_pos) else np.asarray(X1_pos)
        X2_pos = data_dict['adata_omics2'].X
        X2_pos = X2_pos.toarray() if scipy.sparse.issparse(X2_pos) else np.asarray(X2_pos)
        connectivities1 = connectivity(np.where(X1_pos > 0, X1_pos, 0), 1)
        connectivities2 = connectivity(np.where(X2_pos > 0, X2_pos, 0), 1)
        connectivities = list(set(connectivities1).intersection(set(connectivities2)))
    else:
        connectivities = None

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = int(data_dict['adata_omics1'].obs["ground_truth"].dropna().nunique())
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = n_ground_truth

    best_ari = -1.0
    best_seed = None
    best_adata = None

    for seed_val in SEEDS:
        print("\n" + "-" * 60)
        print(f" Dataset: {dataset_name} | Seed: {seed_val} ".center(60, "-"))
        print("-" * 60)

        seed_start_time = time.time()

        # 1. Model Training
        model = SCIGMA(
            data_dict,
            seed_num=seed_val,
            device=device,
            dim_output=dim_output,
            weight_decay=1e-2,
            batch_size=batch_size,
            recon_weight1=1,
            recon_weight2=1,
            contrastive_weight1=contrastive_weight1,
            contrastive_weight2=contrastive_weight2,
            clr_weight=0.3,
            learning_rate=1e-3,
            downweight=0,
            connectivities=connectivities,
            architecture=args.architecture,
        )
        output = model.train(epochs)

        # 2. Latent Representation & Clustering (exact from SCIGMA_SPOTS.py & SCIGMA_ATAC.py)
        adata = data_dict['adata_omics1'].copy()
        adata.obsm['emb_latent_omics1'] = output['emb_latent_omics1']
        adata.obsm['emb_latent_omics2'] = output['emb_latent_omics2']
        adata.obsm['emb_combined'] = output['emb_latent_combined']
        adata.obsm['SCIGMA'] = output['emb_latent_combined']
        adata.obsm['emb_recon_omics1'] = output['emb_recon_omics1']
        adata.obsm['emb_recon_omics2'] = output['emb_recon_omics2']
        adata.obsm['attention'] = output['attention']
        adata.obs['invtau'] = output['invtau']
        adata.obs['tau'] = output['tau']

        if args.tool == 'mclust':
            clustering(
                adata,
                key='emb_combined',
                add_key='SCIGMA',
                n_clusters=n_cluster,
                method='mclust',
                use_pca=False,
                random_seed=1000,
            )
        else:
            clustering(
                adata,
                key='emb_combined',
                add_key='SCIGMA',
                n_clusters=n_cluster,
                method=args.tool,
                start=0.1,
                end=2.0,
                increment=0.01,
                use_pca=False,
            )

        duration_sec = round(time.time() - seed_start_time, 2)

        # 3. Score Generating (8 metrics + Duration, masked by valid_mask)
        if 'anno' not in adata.obs.columns and 'ground_truth' in adata.obs.columns:
            adata.obs['anno'] = adata.obs['ground_truth']

        valid_mask = adata.obs['ground_truth'].notna()
        y_true = adata.obs.loc[valid_mask, 'ground_truth'].astype(str)
        y_pred_eval = adata.obs.loc[valid_mask, 'SCIGMA'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred_eval)
        nmi = normalized_mutual_info_score(y_true, y_pred_eval)
        ami = adjusted_mutual_info_score(y_true, y_pred_eval)
        homogeneity = homogeneity_score(y_true, y_pred_eval)
        v_measure = v_measure_score(y_true, y_pred_eval)

        joint_feat = adata.obsm['emb_combined'][valid_mask.values]
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
            best_adata.obsm[f'{MODEL_NAME}_emb'] = best_adata.obsm['emb_combined'].copy()
            best_adata.obs[f'{MODEL_NAME}_domain'] = best_adata.obs['SCIGMA'].copy()
            best_adata.obs['domain'] = best_adata.obs['SCIGMA'].copy()
            best_adata.uns['selected_seed'] = best_seed

        # ------------------ Plotting & Visualization ------------------
        fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
        sc.pp.neighbors(adata, use_rep='emb_combined', n_neighbors=10)
        sc.tl.umap(adata)

        sc.pl.umap(adata, color='SCIGMA', ax=ax_list[0], title='SCIGMA', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='SCIGMA', ax=ax_list[1], title='SCIGMA', s=20, show=False)
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
# 8. GLOBAL SUMMARY & PUBLICATION VISUALIZATION SUITE
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
