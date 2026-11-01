
# ------------------------ Load Packages ---------------------------------------
import os
import torch
import pandas as pd
import scanpy as sc
import warnings
from muon import prot as pt
from muon import atac as ac




import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score


# ------------------------ MNN ---------------------------------------
import os
import random
import numba as nb
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from torch.backends import cudnn
from tqdm import tqdm


@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def fastSort32(a):
    """
    Perform fast argsort for float32 arrays using Numba parallelization.

    Parameters
    ----------
    a : np.ndarray (float32)
        2D array of distances.

    Returns
    -------
    b : np.ndarray (int32)
        Indices that would sort each row of `a`.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort64(a):
    """
    Perform fast argsort for float64 arrays using Numba parallelization.

    Parameters
    ----------
    a : np.ndarray (float64)
        2D array of distances.

    Returns
    -------
    b : np.ndarray (int32)
        Indices that would sort each row of `a`.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


def Mutual_Nearest_Neighbors(adata, key=None, n_nearest_neighbors=1, farthest_ratio=0.5, max_samples=20000):
    """
    Find mutual nearest neighbors (MNNs) and construct triplets with optional sampling.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    key : str, optional
        Key in `adata.obsm` to use as features (default: use `adata.X`).
    n_nearest_neighbors : int, default=1
        Number of nearest neighbors to consider.
    farthest_ratio : float, default=0.5
        Fraction of farthest neighbors to consider when sampling negatives.
    max_samples : int, default=20000
        Maximum number of cells to process. If dataset is larger, random sampling is applied.

    Returns
    -------
    anchors : list[int]
        Indices of anchor points in original `adata`.
    positives : list[int]
        Indices of positive samples (MNNs).
    negatives : list[int]
        Indices of negative samples (randomly sampled farthest neighbors).
    """
    original_indices = np.arange(adata.shape[0])  # Store original indices
    l = adata.shape[0]

    # Apply sampling if dataset is too large
    if l > max_samples:
        print(f"Dataset size {l} exceeds max_samples {max_samples}, performing random sampling...")
        np.random.seed(42)
        sample_idx = np.random.choice(l, max_samples, replace=False)
        adata_sampled = adata[sample_idx].copy()
        original_indices = original_indices[sample_idx]
        l = max_samples
        print(f"Working with sampled {l} cells")
    else:
        adata_sampled = adata.copy()

    X = adata_sampled.X if key is None else adata_sampled.obsm[key]
    distances = pairwise_distances(X)
    same_count = (distances == 0).sum(axis=1)
    print('Distances calculation completed!')

    nearest_neighbors_index = []
    farthest_neighbors_index = []

    if distances.dtype == "float64":
        sorted_neighbors_index = fastSort64(distances)
    elif distances.dtype == "float32":
        sorted_neighbors_index = fastSort32(distances)

    for i, j in enumerate(same_count):
        nearest_neighbors_index.append(sorted_neighbors_index[i, j:j + n_nearest_neighbors])
        farthest_neighbors_index.append(
            sorted_neighbors_index[i, np.random.choice(
                np.arange(-int((l - j) * farthest_ratio), 0),
                n_nearest_neighbors ** 2
            )]
        )

    nn_dict = {i: set(nearest_neighbors_index[i]) for i in range(l)}

    anchors, positives, negatives = [], [], []
    for i, (nearest_neighbors, farthest_neighbors) in enumerate(zip(nearest_neighbors_index, farthest_neighbors_index)):
        if not np.all(X[i] == 0):
            for j in nearest_neighbors:
                if i in nn_dict[j]:
                    anchors.append(original_indices[i])
                    positives.append(original_indices[j])
                    negatives.append(original_indices[np.random.choice(farthest_neighbors, 1)[0]])

    print(f"The data using feature '{key if key else 'X'}' contains {len(anchors)} mnn_anchors")
    return anchors, positives, negatives


def MMN_batch(X, batches, far_frac=0.6, top_k=2, random_state=None, verbose=True):
    """
    Compute mutual nearest neighbors across batches and construct triplets.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_cells, n_features).
    batches : np.ndarray
        Batch labels of length `n_cells`.
    far_frac : float, default=0.6
        Fraction of farthest same-batch neighbors to consider as negatives.
    top_k : int, default=2
        Number of nearest neighbors to consider between batches.
    random_state : int, optional
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    anchors : np.ndarray
        Anchor indices.
    positives : np.ndarray
        Positive (MNN) indices.
    negatives : np.ndarray
        Negative sample indices.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    X = np.asarray(X)
    batches = np.asarray(batches)

    anchors, positives, negatives = [], [], []
    unique_batches = np.unique(batches)
    batch_to_indices = {b: np.where(batches == b)[0] for b in unique_batches}

    for batch_a, batch_b in combinations(unique_batches, 2):
        idx_a = batch_to_indices[batch_a]
        idx_b = batch_to_indices[batch_b]
        Xa, Xb = X[idx_a], X[idx_b]

        # Nearest neighbor search (a -> b)
        nn_ab = NearestNeighbors(n_neighbors=top_k).fit(Xb)
        _, indices_ab = nn_ab.kneighbors(Xa)
        a2b = {idx_a[i]: set(idx_b[j] for j in indices_ab[i]) for i in range(len(idx_a))}

        # Nearest neighbor search (b -> a)
        nn_ba = NearestNeighbors(n_neighbors=top_k).fit(Xa)
        _, indices_ba = nn_ba.kneighbors(Xb)
        b2a = {idx_b[i]: set(idx_a[j] for j in indices_ba[i]) for i in range(len(idx_b))}

        # Build MNN pairs
        mnn_pairs = [(a, b) for a in a2b for b in a2b[a] if a in b2a.get(b, set())]

        # Construct triplets
        for anchor_idx, pos_idx in mnn_pairs:
            same_batch_idxs = batch_to_indices[batches[anchor_idx]]
            same_batch_idxs = same_batch_idxs[same_batch_idxs != anchor_idx]
            if same_batch_idxs.size == 0:
                continue

            dists = np.linalg.norm(X[same_batch_idxs] - X[anchor_idx], axis=1)
            num_far = max(1, int(len(dists) * far_frac))
            far_indices = same_batch_idxs[np.argpartition(dists, -num_far)[-num_far:]]
            neg_idx = np.random.choice(far_indices)

            anchors.append(anchor_idx)
            positives.append(pos_idx)
            negatives.append(neg_idx)

        if verbose:
            print(f"Batch {batch_a} vs {batch_b}: {len(mnn_pairs)} MNN (top-{top_k}) triplets")

    return np.array(anchors), np.array(positives), np.array(negatives)


# -------------------------------------- Build Graph ---------------------------------------

import numpy as np
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph

def Cal_Spatial_Net(adata, radius=None, n_neighbors=None, model='KNN', verbose=True, include_self=False):
    """
    Construct spatial neighbor graph from spatial coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cell-level spatial coordinates in `adata.obsm['spatial']`.
    radius : float, optional
        Radius for neighborhood search (used when `model='Radius'`).
    n_neighbors : int, optional
        Number of neighbors (used when `model='KNN'`).
    model : {'KNN', 'Radius'}, default='KNN'
        Type of graph construction method.
    verbose : bool, default=True
        If True, print summary of constructed graph.
    include_self : bool, default=False
        Whether to include self-loops in the adjacency matrix.

    Returns
    -------
    None
        The adjacency matrix is stored in `adata.uns['adj']`, and edges in `adata.uns['edgeList']`.
    """
    spatial = adata.obsm['spatial']
    if model == 'KNN':
        adata.uns['adj'] = kneighbors_graph(spatial, n_neighbors=n_neighbors, mode='connectivity', include_self=include_self)
    elif model == 'Radius':
        adata.uns['adj'] = radius_neighbors_graph(spatial, radius=radius, mode='connectivity', include_self=include_self)

    edgeList = np.nonzero(adata.uns['adj'])
    adata.uns['edgeList'] = np.array([edgeList[0], edgeList[1]])

    if verbose:
        print('The graph contains %d edges, %d cells.' % (adata.uns['edgeList'].shape[1], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['edgeList'].shape[1] / adata.n_obs))


# -------------------------------------- Utils ---------------------------------------

import os
import random
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from torch.backends import cudnn
# from harmony import harmonize
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def set_seed(seed=2024):
    """
    Set random seed for reproducibility across Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int, default=2024
        Random seed value.

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def harmony(adata, feature_labels, batch_labels, use_gpu=True):
    """
    Perform batch correction using Harmony.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing features in `.obsm`.
    feature_labels : str
        Key in `adata.obsm` for feature representation.
    batch_labels : str
        Key in `adata.obs` for batch labels.
    use_gpu : bool, default=True
        Whether to use GPU acceleration.

    Returns
    -------
    None
        Updates `adata.obsm` with corrected representation: `{feature_labels}_harmony`.
    """
    df_batches = pd.DataFrame(np.reshape(adata.obs[batch_labels], (-1, 1)), columns=['batch'])
    bc_latent = harmonize(
        adata.obsm[feature_labels],
        df_batches,
        batch_key="batch",
        use_gpu=use_gpu,
        verbose=True,
        max_iter_harmony=20,
        theta=10,
    )
    adata.obsm[f"{feature_labels}_harmony"] = bc_latent


def pca(adata, use_reps=None, n_comps=10):
    """
    Perform dimensionality reduction using PCA.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    use_reps : str, optional
        Key in `adata.obsm` to use as input features.
        If None, use `adata.X`.
    n_comps : int, default=10
        Number of principal components.

    Returns
    -------
    np.ndarray
        PCA-reduced features of shape [n_samples, n_comps].
    """
    from sklearn.decomposition import PCA
    from scipy.sparse import csc_matrix, csr_matrix

    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import numpy as np
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    import pandas as pd

    np.random.seed(random_seed)

    robjects.r.library("mclust")

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)

    rmclust = robjects.r["Mclust"]

    # Get the data
    X = np.array(adata.obsm[used_obsm], dtype=np.float64)

    print("Input shape:", X.shape)

    # Convert to DataFrame with column names
    df = pd.DataFrame(
        X,
        columns=[f'PC{i+1}' for i in range(X.shape[1])]
    )

    # Use subset for initialization to dramatically speed up mclust for large datasets
    subset_size = min(300, X.shape[0])
    subset_indices = robjects.IntVector(list(np.random.choice(range(1, X.shape[0] + 1), subset_size, replace=False)))
    init_list = robjects.ListVector({'subset': subset_indices})

    with localconverter(default_converter + pandas2ri.converter):
        res = rmclust(
            df,
            G=num_cluster,
            modelNames=modelNames,
            initialization=init_list
        )

    # mclust_res = np.array(res.rx2("classification"))
    # mclust_res = np.array(res.getbyname("classification"))
    # Correct for new rpy2 versions
    mclust_res = np.array(res['classification'])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('str')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="SMART",
    method="SMART",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
):
    """
    Perform clustering on latent representations with multiple supported methods.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object of scanpy.
    n_clusters : int, default=7
        Number of clusters.
    key : str, default="emb"
        Key of input representation in `adata.obsm`.
    add_key : str, default="SMART"
        Key to store clustering results in `adata.obs`.
    method : str, default="SMART"
        Clustering method. Options: ["mclust", "leiden", "louvain", "gmm", "kmeans"].
    start : float, default=0.1
        Start resolution for search (used in leiden/louvain).
    end : float, default=3.0
        End resolution for search (used in leiden/louvain).
    increment : float, default=0.01
        Step size for resolution search.
    use_pca : bool, default=False
        Whether to reduce dimensions using PCA.
    n_comps : int, default=20
        Number of components for PCA if `use_pca=True`.

    Returns
    -------
    None
        Updates `adata.obs[add_key]` with clustering results.
    """
    if use_pca:
        adata.obsm[key + "_pca"] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == "mclust":
        adata = mclust_R(adata, used_obsm=(key + "_pca" if use_pca else key), num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs["mclust"]

    elif method in ["leiden", "louvain"]:
        res = search_res(adata, n_clusters, use_rep=(key + "_pca" if use_pca else key),
                         method=method, start=start, end=end, increment=increment)
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["leiden"]
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["louvain"]

    elif method == "gmm":
        X = adata.obsm[key + "_pca"] if use_pca else adata.obsm[key]
        gmm = GaussianMixture(n_components=n_clusters, random_state=0)
        adata.obs[add_key] = gmm.fit_predict(X).astype("category")

    elif method == "kmeans":
        X = adata.obsm[key + "_pca"] if use_pca else adata.obsm[key]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        adata.obs[add_key] = kmeans.fit_predict(X).astype("category")

    else:
        raise ValueError("Clustering method must be one of ['mclust', 'leiden', 'louvain', 'gmm', 'kmeans'].")


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01):
    """
    Search for resolution value that yields the desired number of clusters.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    n_clusters : int
        Target number of clusters.
    method : str, default="leiden"
        Clustering method. Options: ["leiden", "louvain"].
    use_rep : str, default="emb"
        Representation key for clustering.
    start : float, default=0.1
        Start resolution.
    end : float, default=3.0
        End resolution.
    increment : float, default=0.01
        Resolution step size.

    Returns
    -------
    res : float
        Resolution value that yields `n_clusters` clusters.
    """
    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = adata.obs["leiden"].nunique()
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = adata.obs["louvain"].nunique()
        print(f"resolution={res}, cluster number={count_unique}")
        if count_unique == n_clusters:
            return res

    raise ValueError("Resolution not found. Try a bigger range or smaller step size.")


def getcolordict(adata, my_cluster, true_cluster, colordict):
    """
    Map predicted clusters to true clusters using color dictionary.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with clustering results.
    my_cluster : str
        Column name of predicted clusters in `adata.obs`.
    true_cluster : str
        Column name of true clusters in `adata.obs`.
    colordict : dict
        Dictionary mapping true clusters to colors.

    Returns
    -------
    dict
        Mapping from predicted cluster IDs to colors.
    """
    v = adata.obs[[my_cluster, true_cluster]].value_counts()
    colordict1 = {}
    for a in v.index:
        if a[0] not in colordict1.keys() and colordict[a[1]] not in colordict1.values():
            colordict1[a[0]] = colordict[a[1]]

    for a in adata.obs[my_cluster].unique():
        if a not in colordict1.keys():
            print(a)
            for b in colordict.values():
                if b not in colordict1.values():
                    colordict1[a] = b
    return colordict1



# -------------------------------------- Layer ---------------------------------------

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import (
    GATConv, GCNConv, GPSConv, GINConv,
    SAGEConv, LGConv, GATv2Conv, GCN2Conv, GraphConv
)

class SAGEConv_Encoder(torch.nn.Module):
    """
    Encoder based on GraphSAGE convolution.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape [num_nodes, in_channels].
        edge_index : torch.LongTensor
            Graph connectivity in COO format with shape [2, num_edges].

        Returns
        -------
        torch.Tensor
            Encoded node features of shape [num_nodes, out_channels].
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    """
    Decoder based on GraphSAGE convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Encoder(torch.nn.Module):
    """
    Encoder based on GCN convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels, normalize=True)
        self.conv2 = GCNConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCNConv_Decoder(torch.nn.Module):
    """
    Decoder based on GCN convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GCNConv_Decoder, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels, normalize=True)
        self.conv2 = GCNConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Encoder(torch.nn.Module):
    """
    Encoder based on GCNII (GCN2Conv).
    """
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Encoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, out_channels, normalize=True)
        self.conv2 = GCN2Conv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GCN2Conv_Decoder(torch.nn.Module):
    """
    Decoder based on GCNII (GCN2Conv) for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GCN2Conv_Decoder, self).__init__()
        self.conv1 = GCN2Conv(in_channels, in_channels, normalize=True)
        self.conv2 = GCN2Conv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Encoder(torch.nn.Module):
    """
    Encoder based on Graph Attention Networks (GAT).
    """
    def __init__(self, in_channels, out_channels):
        super(GATConv_Encoder, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=2, concat=False)
        self.conv2 = GATConv(out_channels, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GATConv_Decoder(torch.nn.Module):
    """
    Decoder based on GAT convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GATConv_Decoder, self).__init__()
        self.conv1 = GATConv(in_channels, in_channels, heads=2, concat=False)
        self.conv2 = GATConv(in_channels, out_channels, heads=2, concat=False)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GraphConv_Encoder(torch.nn.Module):
    """
    Encoder based on GraphConv.
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Encoder, self).__init__()
        self.conv1 = GraphConv(in_channels, out_channels)
        self.conv2 = GraphConv(out_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the encoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class GraphConv_Decoder(torch.nn.Module):
    """
    Decoder based on GraphConv for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(GraphConv_Decoder, self).__init__()
        self.conv1 = GraphConv(in_channels, in_channels)
        self.conv2 = GraphConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the decoder.
        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x



# -------------------------------------- Model ---------------------------------------
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class SMART(torch.nn.Module):
    """
    SMART: A modular multi-modal graph representation learning model.

    This model uses an encoder-decoder architecture for each modality,
    projects the learned embeddings into a shared latent space, and reconstructs
    input features for self-supervised training.

    Parameters
    ----------
    hidden_dims : list of int
        List specifying input dimensions of each modality and shared hidden dimension.
        Example: [in_dim_mod1, in_dim_mod2, ..., latent_dim].
    device : torch.device
        Device to place model modules on.
    Conv_Encoder : class
        Encoder architecture (default: SAGEConv_Encoder).
    Conv_Decoder : class
        Decoder architecture (default: SAGEConv_Decoder).
    """
    def __init__(self, hidden_dims, device, Conv_Encoder=SAGEConv_Encoder, Conv_Decoder=SAGEConv_Decoder):
        super(SMART, self).__init__()
        out_dim = hidden_dims[-1]

        # One encoder per modality
        self.encoders = [Conv_Encoder(in_dim, out_dim).to(device) for in_dim in hidden_dims[:-1]]
        self.fc = nn.Linear((len(hidden_dims) - 1) * out_dim, out_dim)

        # One decoder per modality
        self.decoders = [Conv_Decoder(out_dim, in_dim).to(device) for in_dim in hidden_dims[:-1]]

    def forward(self, features, edge_indexs):
        """
        Forward pass of the SMART model.

        Parameters
        ----------
        features : list of torch.Tensor
            Node features for each modality. Each tensor shape: [num_nodes, in_dim_mod].
        edge_indexs : list of torch.LongTensor
            Graph connectivity for each modality. Each tensor shape: [2, num_edges].

        Returns
        -------
        z : torch.Tensor
            Latent shared representation of shape [num_nodes, latent_dim].
        x_rec : list of torch.Tensor
            Reconstructed features for each modality.
        """
        # Encode each modality
        x = [encoder(feature, edge_index) for encoder, feature, edge_index in zip(self.encoders, features, edge_indexs)]

        # Concatenate or directly project
        if len(x) == 1:
            z = self.fc(x[0])
        else:
            z = self.fc(torch.cat(x, dim=1))

        # Decode each modality
        x_rec = [decoder(z, edge_index) for decoder, feature, edge_index in zip(self.decoders, features, edge_indexs)]
        return z, x_rec




# -------------------------------------- Train ---------------------------------------

from scipy import stats
import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def laplacian_regularization(x, edge_index):
    """
    Compute Laplacian regularization loss.

    This loss enforces smoothness by penalizing differences between
    embeddings of adjacent nodes.

    Parameters
    ----------
    x : torch.Tensor
        Node embeddings of shape [num_nodes, emb_dim].
    edge_index : torch.LongTensor
        Graph connectivity in COO format with shape [2, num_edges].

    Returns
    -------
    torch.Tensor
        Scalar Laplacian regularization loss.
    """
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss


def train_SMART(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
):
    """
    Train the SMART model with reconstruction, triplet, and optional Laplacian loss.

    Parameters
    ----------
    features : list of torch.Tensor
        Node feature matrices for each modality. Each element has shape [num_nodes, in_dim].
    edges : list of torch.LongTensor
        Graph connectivity for each modality. Each element has shape [2, num_edges].
    triplet_samples_list : list of tuple
        Each tuple contains (anchors, positives, negatives) indices for triplet loss.
    weights : list of float, default=[1, 1, 1, 1]
        Loss weights in the following order (matching the implementation):
        [reconstruction_loss_modality1, reconstruction_loss_modality2,
        triplet_loss_modality1,        triplet_loss_modality2].

        - reconstruction_loss_modalityX: weight for reconstruction (MSE) loss of modality X
        - triplet_loss_modalityX: weight for triplet loss of modality X

        Notes
        -----
        Let M be the number of modalities (len(features)).
        `weights` should have length 2*M:
            weights[0:M]     -> reconstruction weights for each modality
            weights[M:2*M]   -> triplet weights for each modality
    emb_dim : int, default=64
        Dimension of shared latent embedding.
    n_epochs : int, default=500
        Number of training epochs.
    lr : float, default=0.0001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay for optimizer.
    device : torch.device, optional
        Device to train on (default: GPU if available).
    window_size : int, default=20
        Window size for early stopping slope detection.
    slope : float, default=0.0001
        Minimum absolute slope threshold for continuing training.
    Conv_Encoder : class, default=SAGEConv_Encoder
        Graph encoder class.
    Conv_Decoder : class, default=SAGEConv_Decoder
        Graph decoder class.
    margin : float, default=0.5
        Margin for triplet loss.
    return_loss : bool, default=False
        Whether to return loss history along with trained model.
    laplacian_alpha : float, default=0
        Weight for Laplacian regularization term. Disabled if set to 0.

    Returns
    -------
    model : SMART
        Trained SMART model.
    loss_list : list of tuple, optional
        Only returned if `return_loss=True`. Contains (total_loss, tri_loss, rec_loss) per epoch.
    """
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]
    model = SMART(hidden_dims=hidden_dims, device=device,
                  Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features, edges = [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, x_rec = model(features, edges)

        # Triplet loss
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        # Reconstruction loss
        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        # Total loss
        loss = rec_loss + tri_loss

        # Add Laplacian regularization if enabled
        if laplacian_alpha != 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        # Early stopping based on slope of recent loss trend
        if epoch > window_size and epoch % 10 == 0:
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [i[1] for i in loss_list[-window_size:]])  # tri_loss trend
            res2 = stats.linregress(x_axis, [i[2] for i in loss_list[-window_size:]])  # rec_loss trend
            if abs(res1.slope) < slope or abs(res2.slope) < slope:
                if res1.slope != 0 and res2.slope != 0:
                    print("Early stopping: flat trend detected.")
                    break

        # Backward & optimize
        loss_list.append((loss.item(), tri_loss.item(), rec_loss.item()))
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)

def train_SMART_MS(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
):
    """
    Train the SMART-MS model with reconstruction, triplet, and optional Laplacian loss.

    Parameters
    ----------
    features : list of torch.Tensor
        Node feature matrices for each modality. Each element has shape [num_nodes, in_dim].
    edges : list of torch.LongTensor
        Graph connectivity for each modality. Each element has shape [2, num_edges].
    triplet_samples_list : list of tuple
        Each tuple contains (anchors, positives, negatives) indices for triplet loss.
    weights : list of float, default=[1, 1, 1, 1]
        Loss weights in the following order (matching the implementation):
        [reconstruction_loss_modality1, reconstruction_loss_modality2,
        triplet_loss_modality1,        triplet_loss_modality2].

        - reconstruction_loss_modalityX: weight for reconstruction (MSE) loss of modality X
        - triplet_loss_modalityX: weight for triplet loss of modality X

        Notes
        -----
        Let M be the number of modalities (len(features)).
        `weights` should have length 2*M:
            weights[0:M]     -> reconstruction weights for each modality
            weights[M:2*M]   -> triplet weights for each modality
    emb_dim : int, default=64
        Dimension of shared latent embedding.
    n_epochs : int, default=500
        Number of training epochs.
    lr : float, default=0.0001
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        Weight decay for optimizer.
    device : torch.device, optional
        Device to train on (default: GPU if available).
    window_size : int, default=20
        Window size for early stopping slope detection.
    slope : float, default=0.0001
        Minimum absolute slope threshold for continuing training.
    Conv_Encoder : class, default=SAGEConv_Encoder
        Graph encoder class.
    Conv_Decoder : class, default=SAGEConv_Decoder
        Graph decoder class.
    margin : float, default=0.5
        Margin for triplet loss.
    return_loss : bool, default=False
        Whether to return loss history along with trained model.
    laplacian_alpha : float, default=0
        Weight for Laplacian regularization term. Disabled if set to 0.

    Returns
    -------
    model : SMART
        Trained SMART model.
    loss_list : list of tuple, optional
        Only returned if `return_loss=True`. Contains (total_loss, tri_loss, rec_loss) per epoch.
    """
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]
    model = SMART(hidden_dims=hidden_dims, device=device,
                  Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features, edges = [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, x_rec = model(features, edges)

        # Triplet loss
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        # Reconstruction loss
        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        # Total loss
        loss = rec_loss + tri_loss

        # Add Laplacian regularization if enabled
        if laplacian_alpha != 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        # Early stopping based on slope of recent loss trend
        if epoch > window_size and epoch % 10 == 0:
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [i[1] for i in loss_list[-window_size:]])  # tri_loss trend
            res2 = stats.linregress(x_axis, [i[2] for i in loss_list[-window_size:]])  # rec_loss trend
            if abs(res1.slope) < slope or abs(res2.slope) < slope:
                if res1.slope != 0 and res2.slope != 0:
                    print("Early stopping: flat trend detected.")
                    break

        # Backward & optimize
        loss_list.append((loss.item(), tri_loss.item(), rec_loss.item()))
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)



# -------------------------------------- Load Data ---------------------------------------





# warnings.filterwarnings('ignore')
# Environment configuration. SMART pacakge can be implemented with either CPU or GPU. GPU acceleration is highly recommend for imporoved efficiency.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# -------------------------------------- Run Configuration ---------------------------------------

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

# Select which datasets to run (indices 0 to 5)
DATASET_INDICES = [0, 1, 2, 3, 4, 5]

# Define seeds to loop over
SEEDS = [
    13,
    2560,
    641,
    1892,
    1173,
    69,
    2024,
    231,
    1971,
    2497,
    338,
    3127,
    2001,
    2022,
    574,
    2428,
    999,
    1187,
    42,
    3999
]

# Set R_HOME and path before any rpy2 operations
import os
os.environ['R_HOME'] = '/usr/lib/R'
os.environ['PATH'] = '/usr/lib/R/bin:' + os.environ['PATH']

# Import rpy2 and setup mclust
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri, default_converter
import rpy2.robjects.conversion as cv

cv.set_conversion(default_converter + pandas2ri.converter)

# Suppress warnings
robjects.r.options(warn=-1)

# Install mclust only if it is not already installed
robjects.r("""
if (!requireNamespace("mclust", quietly = TRUE)) {
    install.packages("mclust", repos="https://cloud.r-project.org")
}
library(mclust)
""")

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

all_results = []
tool = 'mclust'  # mclust, leiden, and louvain

for dataset_idx in DATASET_INDICES:
    dataset_name, folder_url = choices[dataset_idx]

    print("\n" + "#"*80)
    print(f" STARTING DATASET: {dataset_name} ".center(80, "#"))
    print("#"*80)

    # ------------------ Download & Path Configuration ------------------
    base = f"data/{dataset_name}"
    os.makedirs(base, exist_ok=True)

    rna_path = os.path.join(base, "adata_RNA.h5ad")

    # Human datasets have ADT; Mouse datasets have ATAC
    if dataset_name.startswith("10x"):
        other_path = os.path.join(base, "adata_ADT.h5ad")
        annotation_path = os.path.join(base, "annotation.csv")
        gt_column = "manual-anno"
        data_type = '10x'
    else:
        other_path = os.path.join(base, "adata_ATAC.h5ad")
        annotation_path = os.path.join(base, "anno.csv")
        gt_column = "cluster"
        data_type = 'Spatial-epigenome-transcriptome'

    # Check if files already exist locally before triggering download
    if not os.path.exists(rna_path) or not os.path.exists(other_path) or not os.path.exists(annotation_path):
        print(f"Downloading dataset files into: {base}")
        os.system(f'gdown --folder "{folder_url}" --output "{base}"')
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
    # RNA preprocessing
    sc.pp.filter_genes(adata_omics1, min_cells=10)
    sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
    adata_omics1.obsm['feat'] = pca(adata_omics1_high, n_comps=30)

    # Protein or ATAC preprocessing
    if dataset_name.startswith("10x"):
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        pt.pp.clr(adata_omics2)
        sc.pp.scale(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=30)
    else:
        adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
        ac.pp.tfidf(adata_omics2, scale_factor=1e4)
        sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2)
        adata_omics2.obsm['feat'] = pca(adata_omics2, n_comps=60)

    # ------------------ Spatial Neighbor Graph Construction ------------------
    Cal_Spatial_Net(adata_omics1, model="KNN", n_neighbors=6)
    Cal_Spatial_Net(adata_omics2, model="KNN", n_neighbors=6)

    adata_list = [adata_omics1, adata_omics2]
    x = [torch.FloatTensor(adata.obsm["feat"]).to(device) for adata in adata_list]
    edges = [torch.LongTensor(adata.uns["edgeList"]).to(device) for adata in adata_list]

    # ------------------ Seed Iterations ------------------
    dataset_results = []
    n_ground_truth = adata_omics1.obs["ground_truth"].nunique()
    print(f"Number of ground truth classes: {n_ground_truth}")
    n_cluster = n_ground_truth

    for seed in SEEDS:
        print(f"\n" + "-"*60)
        print(f" Dataset: {dataset_name} | Seed: {seed} ".center(60, "-"))
        print("-"*60)

        # 1. Set seed
        set_seed(seed)

        # 2. Triplet Sample calculation
        triplet_samples_list = [
            Mutual_Nearest_Neighbors(adata, key="feat", n_nearest_neighbors=3, farthest_ratio=0.6)
            for adata in adata_list
        ]

        # 3. Model Training
        model = train_SMART(
            features=x,
            edges=edges,
            triplet_samples_list=triplet_samples_list,
            weights=[1, 1, 1, 1],
            emb_dim=64,
            n_epochs=300,
            lr=5e-3,
            weight_decay=1e-6,
            device=device,
            window_size=10,
            slope=1e-4
        )

        # 4. Latent Representation Extract
        adata_omics1.obsm["SMART"] = model(x, edges)[0].cpu().detach().numpy()

        # 5. Clustering
        clustering(adata_omics1, key='SMART', add_key='SMART', n_clusters=n_cluster, method=tool, use_pca=True)

        # 6. Score Generating
        adata = adata_omics1.copy()
        if 'anno' not in adata.obs.columns and 'ground_truth' in adata.obs.columns:
            adata.obs['anno'] = adata.obs['ground_truth']

        y_true = adata.obs['ground_truth'].astype(str)
        y_pred = adata.obs['SMART'].astype(str)

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        homogeneity = homogeneity_score(y_true, y_pred)
        v_measure = v_measure_score(y_true, y_pred)

        # Calculate Silhouette Score
        joint_feat = adata.obsm['SMART']
        sil_score = silhouette_score(
            joint_feat,
            adata.obs['SMART'].astype(int)
        )

        print(f"\nResult for dataset: {dataset_name} | seed: {seed}")
        print(f"ARI: {ari:.4f} | NMI: {nmi:.4f} | Silhouette: {sil_score:.4f}")

        # Store seed results
        res_dict = {
            'dataset': dataset_name,
            'seed': seed,
            'ARI': ari,
            'NMI': nmi,
            'AMI': ami,
            'Homogeneity': homogeneity,
            'V-measure': v_measure,
            'Silhouette': sil_score
        }
        dataset_results.append(res_dict)
        all_results.append(res_dict)

        # ------------------ Plotting & Visualization ------------------
        fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
        sc.pp.neighbors(adata, use_rep='SMART', n_neighbors=10)
        sc.tl.umap(adata)

        sc.pl.umap(adata, color='SMART', ax=ax_list[0], title='SMART', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='SMART', ax=ax_list[1], title='SMART', s=20, show=False)
        sc.pl.embedding(adata, basis='spatial', color='anno', ax=ax_list[2], title='anno', s=20, show=False)
        plt.tight_layout(w_pad=0.3)

        os.makedirs(f"plots/{dataset_name}", exist_ok=True)
        plot_path = f"plots/{dataset_name}/umap_seed_{seed}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

    # Write dataset summary statistics and save csv for this dataset
    df_ds = pd.DataFrame(dataset_results)
    print("\n" + "="*80)
    print(f" RESULTS SUMMARY FOR {dataset_name} ".center(80, "="))
    print("="*80)
    print(df_ds.to_string(index=False))
    print("-"*80)
    print("Summary Statistics:")
    print(df_ds.describe().loc[['mean', 'std']])
    print("="*80)

    # Generate text summary with Mean ± SD format for this dataset
    summary_lines = [f"Summary Stats for {dataset_name}:"]
    metrics_list = ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette']
    for metric in metrics_list:
        mean_val = df_ds[metric].mean()
        std_val = df_ds[metric].std()
        summary_lines.append(f"{metric} = {mean_val:.4f} ± {std_val:.4f}")

    dataset_summary_str = "\n".join(summary_lines)
    print(dataset_summary_str)
    print("="*80)

    os.makedirs("results", exist_ok=True)

    # Save the text summary for this dataset
    with open(f"results/summary_{dataset_name}.txt", "w") as f:
        f.write(dataset_summary_str)

    df_ds.to_csv(f"results/metrics_{dataset_name}.csv", index=False)
    print(f"Saved dataset metrics to 'results/metrics_{dataset_name}.csv'")
    print(f"Saved dataset summary to 'results/summary_{dataset_name}.txt'")

    # Generate publication-quality boxplot + stripplot for this dataset
    plt.figure(figsize=(10, 6))
    df_ds_long = df_ds.melt(
        id_vars=['seed'],
        value_vars=metrics_list,
        var_name='Metric',
        value_name='Score'
    )
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        palette='Set2',
        showfliers=False,
        width=0.5
    )
    sns.stripplot(
        x='Metric',
        y='Score',
        data=df_ds_long,
        color='black',
        alpha=0.6,
        jitter=0.2,
        size=5,
        ax=ax
    )
    plt.title(f"Variability of Metrics across 20 Seeds ({dataset_name})", fontsize=14, pad=15)
    plt.xlabel("Metric", fontsize=12, labelpad=10)
    plt.ylabel("Score", fontsize=12, labelpad=10)
    plt.ylim(0, 1.05)
    plt.tight_layout()

    boxplot_path = f"results/boxplot_{dataset_name}.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Saved variability boxplot to '{boxplot_path}'\n")

# -------------------------------------- Global Summary statistics ---------------------------------------

df_all = pd.DataFrame(all_results)
os.makedirs("results", exist_ok=True)
df_all.to_csv("results/metrics_all_datasets.csv", index=False)

# Compile global text summary
global_summary_lines = [
    "="*80,
    "GLOBAL SUMMARY: MEAN ± STD ACROSS 20 SEEDS",
    "="*80
]

for d_name in df_all['dataset'].unique():
    df_d = df_all[df_all['dataset'] == d_name]
    global_summary_lines.append(f"\nDataset: {d_name}")
    global_summary_lines.append("-" * len(f"Dataset: {d_name}"))
    for metric in ['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette']:
        mean_val = df_d[metric].mean()
        std_val = df_d[metric].std()
        global_summary_lines.append(f"{metric} = {mean_val:.4f} ± {std_val:.4f}")
global_summary_lines.append("\n" + "="*80)

global_summary_str = "\n".join(global_summary_lines)
print(global_summary_str)

summary_stats_path = "results/summary_statistics.txt"
with open(summary_stats_path, "w") as f:
    f.write(global_summary_str)

print("\n" + "#"*80)
print(" GLOBAL RESULTS ACROSS ALL DATASETS ".center(80, "#"))
print("#"*80)
print(df_all.groupby('dataset').mean())
print("#"*80)
print(f"Saved combined metrics to 'results/metrics_all_datasets.csv'")
print(f"Saved global summary statistics to '{summary_stats_path}'")

# Generate comparative global boxplot across all datasets
df_all_long = df_all.melt(
    id_vars=['dataset', 'seed'],
    value_vars=['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette'],
    var_name='Metric',
    value_name='Score'
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
        showfliers=False
    )
    sns.stripplot(
        ax=axes[i],
        x='dataset',
        y='Score',
        data=df_metric,
        color='black',
        alpha=0.5,
        jitter=0.15,
        size=4
    )
    axes[i].set_title(f"{metric} Comparison", fontsize=14)
    axes[i].set_xlabel("Dataset", fontsize=12)
    axes[i].set_ylabel("Score", fontsize=12)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_ylim(0, 1.05)

plt.suptitle("Performance Comparison Across All Datasets & 20 Seeds", fontsize=18, y=0.98)
plt.tight_layout()
global_plot_path = "results/global_boxplot_comparison.png"
plt.savefig(global_plot_path, dpi=300)
plt.close()
print(f"Saved global comparison boxplot to '{global_plot_path}'")

# -------------------------------------- End ---------------------------------------
