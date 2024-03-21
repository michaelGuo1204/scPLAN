import anndata as ad
import numpy as np
import torch
import scanpy as sc
import pandas as pd

from treelib import Tree
from scplan.DataWork.exter.learn import learn_tree
from scplan.DataWork.utils import generate_const_partial_labels, normalize, TrainData, DataModule, \
    createCandidateSet,CellTypeNode

from scplan.deps import logger


class AtlasDataset:
    """
    Dataset module for atlas data
    """

    def __init__(self, **args):
        """
        Initialize the dataset

        :param args: Specify data path and file names with  "data_path" and "target/ref"
        :exception Raise exception if data path/file name is not provided
        """
        if 'data_path' in args.keys():
            self.data_path = args['data_path']
            self.case = 'real'
            try:
                self.target_filename = args['target']
                self.ref_filename = args['ref']
            except:
                raise Exception("Please specify your data domain and file name")
        else:
            raise Exception(
                "Please specify either a data path for real data or simulation case for splatter simulation")
        self.all_data = None  # For target data
        self.ref_cell_dic = None  # Cell type names for each numeric label
        self.all_cell_dic = None  # Cell type for original cell_types
        self.dataset_names = ["ref", "target"]  # Just dataset names
        self.partial_Y = None  # Partial labels
        self.tree = None  # Tree for labels
        self.label_cluster = None  # Label cluster for each label
        self.current_resolution = 1  # Current resolution for label cluster

    def initialize(self, batch_correction=False,tree=None):
        """
        Initialize the dataloader and generate partial label

        :param partial_rate: p to generate false labels among ambiguous pairs
        :except Expection: Raise exception if data file is not found
        :return: partial_Y: partial label for training
                cell_dic: Cell type for each label
        """
        self.readData()
        self.dataset_names = ['{}'.format(self.ref_filename.split('.')[0]),'{}'.format(self.target_filename.split('.')[0])]
        all_data = ad.concat({self.dataset_names[0]: self.ref_data,
                                   self.dataset_names[1]: self.target_data},
                                  axis=0, label='dataset')
        raw_data = all_data.raw.to_adata().copy()
        sc.pp.normalize_per_cell(raw_data)
        all_data.obs['size_factor'] = raw_data.obs.n_counts / np.median(raw_data.obs.n_counts)
        if batch_correction == 'scale':
            sc.pp.scale(all_data)
        elif batch_correction:
            sc.pp.combat(all_data, key='dataset')
        # Labelize the cell type
        if tree != None:
            self.tree = tree
        else:
            self.scHPLTreeBuild(all_data) # Would add tree in self.tree
        tree_cell_dic = [node for node in self.tree.all_nodes() if node.tag != "root"]
        ref_cell_dic = [node for node in self.tree.leaves()]
        branch_ct = set(tree_cell_dic).difference(set(ref_cell_dic))
        branch_dic = {node.tag: [node.tag for node in
                                 self.tree.leaves(node.identifier)] for node in branch_ct}
        self.all_cell_dic = np.array([node.tag for node in (ref_cell_dic + list(branch_ct))])
        self.ref_cell_dic = np.array([node.tag for node in ref_cell_dic])
        branch_ct = [node.tag for node in branch_ct]
        self.candidate_set = createCandidateSet(self.tree, 1)
        self.label_cluster = self.computeLabelCluster(self.candidate_set,self.ref_cell_dic)
        self.all_label_cluster = torch.arange(len(self.ref_cell_dic), len(self.all_cell_dic))
        all_data.obs['cell_label'] = pd.Categorical(all_data.obs['cell_type'].to_numpy(), categories=self.all_cell_dic).codes
        ref_index = all_data.obs['cell_type'].apply(lambda x: x in list(self.ref_cell_dic))
        target_index = all_data.obs['cell_type'].apply(lambda x: x in list(branch_ct))
        ref_partial_Y = self.generateRefLabels(all_data.obs['cell_label'][ref_index],self.ref_cell_dic)
        target_partial_Y = self.generateCandidateLabels(all_data.obs['cell_label'][target_index],self.ref_cell_dic,self.all_cell_dic,branch_dic)
        partial_Y = torch.zeros(len(all_data.obs['cell_label']),len(self.ref_cell_dic))
        partial_Y[ref_index,:] = torch.from_numpy(ref_partial_Y).float()
        partial_Y[target_index,:] = torch.from_numpy(target_partial_Y).float()
        self.partial_Y = partial_Y
        # Generate partial label
        all_data.obsm["partial_label"] = partial_Y
        self.all_data = all_data
        return self.partial_Y, self.ref_cell_dic

    def readData(self):
        if self.case == 'real':
            try:
                self.ref_data = ad.read_h5ad(self.data_path + self.ref_filename)
                self.target_data = ad.read_h5ad(self.data_path + self.target_filename)
            except Exception as e:
                raise Exception("Could not load data")
        if self.case == 'simu':
            try:
                self.all_data = self.generateSimulation()
            except Exception as e:
                raise Exception("Simulation Failed")
    def load(self, batch_size):
        """
        Generate dataloader

        :param batch_size:
        :return: dataloader
        """
        train_dataset = TrainData(DataModule(self.all_data, dropout_p=0.1))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=8,drop_last=True)
        return train_loader

    def scHPLTreeBuild(self, data):
        classifier = 'svm'
        dimred = True
        dataset_order = np.unique(data.obs['dataset']).tolist()
        schpltree, _ = learn_tree(data=data, cell_type_key="cell_type", batch_key='dataset', batch_order=dataset_order
                                  , classifier=classifier, dimred=dimred)
        self.tree = generateTree(schpltree)

    def generateRefLabels(self,train_labels,label_ref):
        K = int(len(label_ref))
        partialY= np.zeros((len(train_labels),K))
        for i in np.unique(train_labels):
            samples = np.where(train_labels == i)[0]
            partialY[samples, :] = np.repeat(np.eye(K)[i, :].reshape(1, K), len(samples), axis=0)
        return partialY
    def generateCandidateLabels(self, train_label, label_ref,label_all,brach_dic):
        """
        Base function for partial label generation

        :param train_label: Sample label
        :param partial_rate: Probability to generate a false positive within candidate set
        :param label_ref: Cell type corresponding to each label, just for
        :param resolution: Resolution for tree node merge
        :return: partial label and None(For compatibility with Atlas dataset)
        """
        K = int(len(label_ref))
        K_all = int(len(label_all))
        n = int(len(train_label))
        transition_matrix = np.eye(N=K_all,M=K)
        for key,value in brach_dic.items():
            idx_key = np.where(label_all == key)[0][0]
            idx_value = [np.where(label_all == i)[0][0] for i in value]
            transition_matrix[idx_key,idx_value] = 1.0
        partialY = np.zeros((n, K))
        for j in range(n):  # for each instance
            partialY[j, :] = (transition_matrix[train_label[j], :])
        return partialY

    def updateParams(self, param):
        """
        Update params based on input data, numclass, numfeatures and ambiguous cluster would be updated

        :param param: Param() objects
        :return: updated param
        """
        param.num_class = len(self.ref_cell_dic)
        param.num_features = self.all_data.n_vars
        return param

    def computeLabelCluster(self,candidate_set,cell_dic):
        label_cluster = torch.zeros(len(cell_dic))
        for i in range(len(candidate_set)):
            cluster_labels = np.array([np.where(cell_dic == node.tag)[0][0] for node in candidate_set[i]])
            label_cluster[cluster_labels] = i
        print(label_cluster)
        return label_cluster
    def getPartialLabel(self):
        return self.partial_Y
    def getCellDic(self):
        return self.all_cell_dic
    def getRawDataMeta(self):
        return self.all_data.var,self.all_data.obs
    def getLabelCluster(self,regenerate=False,all=False):
        if regenerate:
            self.current_resolution += 1
            if self.current_resolution > self.tree.depth():
                raise Exception("No more levels in the tree")
            self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
            self.label_cluster = self.computeLabelCluster(self.candidate_set,self.ref_cell_dic)
        if all:
            return torch.concat([self.label_cluster,self.all_label_cluster])
        return self.label_cluster
    def getDataIndex(self):
        return self.all_data.obs_names


def generateTree(schpltree) -> Tree:
    tree = Tree()
    tree.create_node(identifier=schpltree[0].name[0], data=CellTypeNode(is_ct=True,type='ROOT'))
    while len(schpltree) > 0:
        node = schpltree.pop(0)
        if len(node.descendants) != 0:
            for child in node.descendants:
                try:
                    tree.create_node(identifier=child.name[0], parent=node.name[0],data=CellTypeNode(is_ct=True,type=child.name[0]))
                    schpltree.append(child)
                except:
                    continue
    tree.show(data_property='type')
    return tree
