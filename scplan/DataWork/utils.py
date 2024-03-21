import numpy as np
import anndata as ad
import scanpy as sc
import torch

from treelib import Tree,Node


class DataModule:
    """
    Wrapper to combine dataforms
    """
    def __init__(self,  values: ad.AnnData,dropout_p:float):
        """
        Initialize the datamodule

        :param values: Main anndata object storing data
        :param dropout_p: Dropout rate for input data for data augmentation
        """
        self.n_obs = values.n_obs
        self.n_vars = values.n_vars
        try:                                                            # Whether data is sparse or not
            self.X = torch.Tensor(values.X)
        except:
            self.X = torch.Tensor(values.X.todense())
        try:
            self.X_raw = torch.Tensor(values.raw.X)
        except:
            self.X_raw = torch.Tensor(values.raw.X.todense())
        mask_elements = int(self.X.shape[1] * dropout_p)                # Construct dropout mask
        mask = torch.rand_like(self.X).argsort(1) >= mask_elements
        self.X_dropout = mask * self.X
        self.size_factor = torch.Tensor(values.obs["size_factor"])      # Compute size factor
        try:
            self.dataset = list(values.obs["dataset"])                  # Specify dataset source
        except:
            self.dataset = ["ref" for i in range(values.n_obs)]
        self.cell_label = torch.Tensor(values.obs["cell_label"]).int()  # Initialize ground truth label
        self.partial_label = torch.Tensor(values.obsm["partial_label"]) # Initialize candidate label

    def __len__(self):
        return self.n_obs
    def __getitem__(self, index):
        """
        Get each observation from index
        :param index:
        :return: X: Normalized data
                X_dropout: Data with dropout
                X_raw: Raw data
                size_factor: Size factor for each observation
                label: Candidate label
                cell_label: Ground truth label
                dataset: Dataset source
                index: Index of observation
        """
        each_obs = self.X[index]
        each_obs_mask = self.X_dropout[index]
        each_obs_raw = self.X_raw[index]
        each_cell_label = self.cell_label[index]
        each_size_factor = self.size_factor[index]
        each_label = self.partial_label[index]
        each_dataset = self.dataset[index]
        return each_obs, each_obs_mask,each_obs_raw, each_size_factor, each_label, each_cell_label,each_dataset, index

class TrainData(torch.utils.data.Dataset):
    """
    Worker training data in PLAN as torch.dataset
    """
    def __init__(self,data:DataModule):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        """
        Get each observation from index

        :param index: Index of data
        :returns:
            - X:Normalized data
            - X_dropout:Data with dropout
            - X_raw: Raw data
            - size_factor: Size factor for each observation
            - label: Candidate label
            - cell_label: Ground truth label
            - dataset: Dataset source
            - index: Index of observation
        """
        return self.data[index]
def generate_uniform_partial_labels(train_labels, label_ref,candidate_set,partial_rate=0.1):
    uncertain_index = []
    for uncertain_group in candidate_set:
        uncertain_labels = [i for i in range(len(label_ref)) if label_ref[i] in uncertain_group]
        uncertain_index.append([[i, j] for i in uncertain_labels for j in uncertain_labels])
    K = int(len(label_ref))
    n = int(len(train_labels))
    transition_matrix = np.eye(K)
    for uncertain_group in uncertain_index:
        for ind in uncertain_group:
            transition_matrix[ind[0], ind[1]] = 1
    partialY = np.zeros((n, K))
    for i in np.unique(train_labels):
        samples = np.where(train_labels == i)[0]
        np.random.shuffle(samples)
        groud_truth = samples[int(len(samples) * partial_rate):]
        partialY[groud_truth, :] = np.repeat(np.eye(K)[i, :].reshape(1, K), len(groud_truth), axis=0)
        missing = samples[:int(len(samples) * partial_rate)]
        trans_i_mat = np.repeat(transition_matrix[i, :].reshape(1, K), len(missing), axis=0)
        partialY[missing, :] = trans_i_mat
    return partialY
def generate_uniform_cv_candidate_labels(train_labels, label_ref,partial_rate=0.1):
    # yjuny:K为类别数量，n为样本数量
    K = int(len(label_ref))
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    # yjuny:保证候选标签集合里面具有真值
    partialY[torch.arange(n), train_labels] = 1.0
    # yjuny:转移矩阵
    transition_matrix = np.eye(K)

    # yjuny:~np.eye(K)表示反转，且dtype=bool把0/1变为False/True np.where(condition)返回满足条件的坐标
    # yjuny:除了真实的标签外，其他混淆标签加入候选标签机和的概率均为partial_rate
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    random_n = np.random.uniform(0, 1, size=(n, K))
    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)
    return partialY
def generate_const_partial_labels(train_labels, label_ref,candidate_set,partial_rate=0.1):
    uncertain_index = []
    for uncertain_group in candidate_set:
        uncertain_labels = [i for i in range(len(label_ref)) if label_ref[i] in uncertain_group]
        uncertain_index.append([[i, j] for i in uncertain_labels for j in uncertain_labels])
    K = int(len(label_ref))
    n = int(len(train_labels))
    partialY = np.zeros((n, K))
    #partialY[np.arange(n), train_labels] = 1
    transition_matrix = np.eye(K)
    for uncertain_group in uncertain_index:
        for ind in uncertain_group:
            transition_matrix[ind[0], ind[1]] = 1
    np.fill_diagonal(transition_matrix, 1)
    for j in range(n):  # for each instance
        partialY[j, :] = (transition_matrix[train_labels[j], :])
    return partialY

class CellTypeNode:
    def __init__(self,type,is_ct):
        self.type = type
        self.is_ct = is_ct


def normalize(adata, copy=True, highly_variable=False, filter_min_counts=False, size_factors=True, normalize_input=False,
              logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if logtrans_input:
        sc.pp.log1p(adata)

    if highly_variable!= False:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_variable,
                                    subset=True)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factor'] = 1.0
    if normalize_input:
        sc.pp.scale(adata)
    return adata


def createCandidateSet(tree, resolution=None):
    """
    Generate candidate set for each cell type

    :param tree:
    :param level_dic:
    :param resolution:
    :return:
    """
    assert resolution <= tree.depth()
    branch_at_level = [node for node in tree.all_nodes() if tree.depth(node) == resolution and not node.is_leaf()]
    leaves_before = [node for node in tree.leaves() if tree.depth(node) <= resolution]
    all_nodes = branch_at_level + leaves_before
    candidate_set = [tree.leaves(node.identifier) for node in all_nodes]
    return candidate_set
