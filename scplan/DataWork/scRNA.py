import anndata as ad
import numpy as np
import scanpy as sc
import torch
import pandas as pd
import scipy
import treelib
from collections import Counter
from anndata.experimental.pytorch import AnnLoader
from typing import Union
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from treelib import Tree
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage,to_tree,inconsistent,fcluster,maxinconsts
from scplan.DataWork.utils import generate_uniform_partial_labels, normalize, TrainData, DataModule,createCandidateSet,CellTypeNode,generate_const_partial_labels,generate_uniform_cv_candidate_labels
from scplan.deps import logger


class scRNADataset:
    """
    Dataset module for single dataset
    """
    def __init__(self, **args):
        """
        :param args: Specify data path and file names with  "data_path" and "target or simulation params "
        :exception Raise exception if neither data path/file name nor simulation params is not provided
        """
        if 'data_path' in args.keys():
            self.data_path = args['data_path']
            self.ref_filename= args['ref']
            self.case = 'real'
        elif 'param' in args.keys():
            self.simulation_param = args['param']
            self.case = 'simu'
        else:
            raise Exception("Please specify either a data path for real data or simulation case for splatter simulation")
        self.all_data = None            # For target data
        self.dataset_names = ["ref"]
        self.ref_cell_dic = None            # Cell type names for each numeric label
        self.partial_Y = None           # Partial labels
        self.tree = None                # Tree for labels
        self.candidate_set = None       # Candidate set for ambiguous pairs

    def initialize(self,partial_rate,resolution=0):
        """
        Initialize the dataloader and generate partial label

        :param partial_rate: p to generate false labels among ambiguous pairs
        :except Exception: Raise exception if data file is not found
        :return: partial_Y: partial label for training
                cell_dic: Cell type for each label
        """
        self.readData()
        # Preprocessing of data
        self.all_data = normalize(self.all_data,highly_variable=False)
        # Get labels from cell_type
        self.all_data.obs["dataset"] = ['ref' for _ in range(self.all_data.n_obs)]
        cell_names = self.all_data.obs["cell_type"]
        self.ref_cell_dic, self.all_data.obs["cell_label"] = np.unique(cell_names, return_inverse=True)
        self.HC_Tree_buildup(self.all_data)
        self.candidate_set = createCandidateSet(self.tree, resolution=0)
        self.partial_Y, _ = self.generateCandidateLabels(self.all_data.obs["cell_label"], partial_rate, self.ref_cell_dic, resolution=resolution)
        self.label_cluster = torch.arange(len(self.ref_cell_dic))
        self.all_data.obsm["partial_label"] = self.partial_Y
        return self.partial_Y,self.ref_cell_dic

    def readData(self):
        if self.case == 'real':
            try:
                self.all_data = ad.read_h5ad(self.data_path+self.ref_filename)
            except Exception as e:
                logger.error("{}".format(e))
                raise Exception("Could not load data")
        if self.case == 'simu':
            try:
                self.all_data = self.generateSimulation()
            except Exception as e:
                logger.error("{}".format(e))
                raise Exception("Simulation Failed")
        pass

    def generateSimulation(self):
        """
        Generate simulated RNASeq counts data

        :return: Andata object for data
        """
        import rpy2.robjects.packages
        splatter = rpy2.robjects.packages.importr("splatter")
        splatter_param = splatter.newSplatParams()
        for k, v in self.simulation_param.items():
            splatter_param = splatter.setParam(splatter_param, k, v)
        sim = splatter.splatSimulateGroups(splatter_param,verbose=False)
        count = rpy2.robjects.r['counts'](sim)
        count_df = rpy2.robjects.r["as.data.frame"](count)
        group = rpy2.robjects.r['colData'](sim)
        group_df = rpy2.robjects.r["as.data.frame"](group)
        from rpy2.robjects import pandas2ri
        with (rpy2.robjects.default_converter + pandas2ri.converter).context():
            count_pd_from_r = rpy2.robjects.conversion.get_conversion().rpy2py(count_df).astype(float)
            group_pd_from_r = rpy2.robjects.conversion.get_conversion().rpy2py(group_df)
        adata = sc.AnnData(count_pd_from_r.T)
        group_pd_from_r.columns = ['cell', 'batch', 'cell_type', "lib_size"]
        adata.obs = group_pd_from_r
        return adata
    def load(self,batch_size,train=True,drop_p=0.1):
        """
        Generate dataloader

        :param batch_size:
        :return: dataloader
        """
        if train:
            '''
            dataset = pd.Categorical(self.all_data.obs["dataset"]).codes
            counts = np.bincount(dataset)
            labels_weights = 1. / counts
            weights = labels_weights[dataset]
            sampler = WeightedRandomSampler(weights, len(dataset))
            '''
            train_dataset = TrainData(DataModule(self.all_data, drop_p))
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                   num_workers=8, drop_last=True,shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=DataModule(self.all_data, drop_p), batch_size=batch_size,num_workers=8)
        return train_loader

    def reloadloader(self,batch_size,mask,drop=0.1):
        ref_mask = np.where(self.all_data.obs["dataset"] == "ref")[0]
        mask = np.array(list(set(ref_mask) | set(mask)))
        sampler = torch.utils.data.sampler.SubsetRandomSampler(mask)
        train_dataset = TrainData(DataModule(self.all_data, drop))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                    num_workers=8, drop_last=True,sampler=sampler)
        return train_loader
    def computePrototype(self,data):
        """
        Compute prototype for each cell type

        :param data: Anndata object
        :return: prototype for each cell type
        """
        prototype_dic = {}
        ref_ct = np.unique(data.obs["cell_type"])
        for celltype in ref_ct:
            type_data = data[data.obs["cell_type"] == celltype]
            type_center = np.array(type_data.X.mean(axis=0)).reshape(-1)
            prototype_dic.update({celltype: type_center})
        prototype = pd.DataFrame(prototype_dic).T.to_numpy()
        return prototype

    def HC_Tree_buildup(self,data):
        '''
        Build up a hierarchical clustering tree based on the mean prototype of each cell type

        :param data: Input anndata object
        :return: A clustering tree with nodes on each level
        '''
        prototype = self.computePrototype(data)
        # Build up a hierarchical clustering tree based on hierarchical clustering
        Z = linkage(prototype, method='ward', metric='euclidean')
        R = inconsistent(Z)
        MI = maxinconsts(Z, R)
        best_cluster = 0
        best_score = -np.inf
        for i in range(2, len(self.ref_cell_dic)):
            clusters = fcluster(Z, i, criterion='maxclust_monocrit', monocrit=MI)
            try:
                score = silhouette_score(prototype, clusters)
                if score > best_score:
                    best_score = score
                    best_cluster = i
                print("Max_t:{},SS:{}".format(i, score))
            except:
                continue
        clusters = fcluster(Z, best_cluster, criterion='maxclust_monocrit', monocrit=MI)
        ref_ct = np.unique(data.obs["cell_type"])
        self.tree = generateTree(clusters,ref_ct)
        return prototype

    def generateCandidateLabels(self, train_label, partial_rate, label_ref,resolution=0):
        """
        Base function for partial label generation

        :param train_label: Sample label
        :param partial_rate: Probability to generate a false positive within candidate set
        :param label_ref: Cell type corresponding to each label, just for
        :param resolution: Resolution for tree node merge
        :return: partial label and None(For compatibility with Atlas dataset)
        """
        active_candidateset = [subset for subset in self.candidate_set if len(subset) > 1]
        active_ct = [[node.tag for node in subset] for subset in active_candidateset]
        print(active_ct)
        return generate_uniform_partial_labels(train_labels=train_label, partial_rate=partial_rate,
                                                    label_ref=label_ref,candidate_set=active_ct), None

    def updateParams(self, param):
        """
        Update params based on input data, numclass, numfeatures and ambiguous cluster would be updated

        :param param: Param() objects
        :return: updated param
        """
        param.num_class = len(self.ref_cell_dic)
        param.num_features = self.all_data.n_vars
        param.debatch = False
        param.ref = {}
        return param
    def getPartialLabel(self):
        return self.partial_Y
    def getCellDic(self):
        return self.ref_cell_dic
    def getRawDataMeta(self):
        return self.all_data.var,self.all_data.obs
    def getLabelCluster(self,regenerate=False,all=False):
        return self.label_cluster
    def getDataIndex(self):
        return self.all_data.obs_names


class HierarchicalDataset(scRNADataset):
    def __init__(self, **args):
        super().__init__(**args)
        assert isinstance(args['tree'], Tree), "Please provide a tree object"
        self.tree = args['tree']
        self.current_resolution = 1
        self.dataset_names = ["ref"]

    def initialize(self, partial_rate, resolution=0):
        self.readData()
        # Preprocessing of data
        self.all_data = normalize(self.all_data, highly_variable=False)
        # Get labels from cell_type
        cell_names = self.all_data.obs["cell_type"]
        self.ref_cell_dic, self.all_data.obs["cell_label"] = np.unique(cell_names, return_inverse=True)
        self.ref_cell_dic = self.checkCellTypeConsist(self.ref_cell_dic,self.tree)
        self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
        self.label_cluster = self.computeLabelCluster(self.candidate_set,self.ref_cell_dic)
        self.partial_Y, _ = self.generateCandidateLabels(self.all_data.obs["cell_label"], partial_rate, self.ref_cell_dic,
                                                         resolution=resolution)
        self.all_data.obsm["partial_label"] = self.partial_Y
        return self.partial_Y, self.ref_cell_dic

    def initialize_novelcell(self,masked_celltype,partial_rate,resolution=0):
        self.readData()
        self.all_data = normalize(self.all_data, highly_variable=False)
        cell_names = self.all_data.obs["cell_type"]
        masked_index = cell_names.apply(lambda x:x in masked_celltype)
        cell_names = self.all_data.obs["cell_type"]
        self.ref_cell_dic, self.all_data.obs["cell_label"] = np.unique(cell_names, return_inverse=True)
        self.ref_cell_dic = self.checkCellTypeConsist(self.ref_cell_dic, self.tree)
        self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
        self.label_cluster = self.computeLabelCluster(self.candidate_set, self.ref_cell_dic)
        self.partial_Y, _ = self.generateCandidateLabels(self.all_data.obs["cell_label"], partial_rate, self.ref_cell_dic,
                                                         resolution=resolution)
        self.partial_Y[masked_index,:] = 1
        self.all_data.obsm["partial_label"] = self.partial_Y
        return self.partial_Y, self.ref_cell_dic


    def checkCellTypeConsist(self,cell_dic,tree):
        ct_from_tree = [node.tag for node in tree.leaves()]
        cell_dic = list(cell_dic)
        if set(cell_dic).difference(set(ct_from_tree))!= set():
            print("Possible Novel Cell Detected")
        cell_dic += list(set(ct_from_tree).difference(set(cell_dic)))
        cell_dic = np.array(cell_dic)
        return cell_dic


    def computeLabelCluster(self,candidate_set,cell_dic):
        label_cluster = torch.zeros(len(cell_dic))
        for i in range(len(candidate_set)):
            cluster_labels = np.array([np.where(cell_dic == node.tag)[0][0] for node in candidate_set[i]])
            label_cluster[cluster_labels] = i
        print(label_cluster)
        return label_cluster

    def getLabelCluster(self,regenerate=False):
        if regenerate:
            self.current_resolution += 1
            if self.current_level > self.tree.depth():
                raise Exception("No more levels in the tree")
            self.candidate_set = createCandidateSet(self.tree, resolution=self.current_level)
            self.label_cluster = self.computeLabelCluster(self.candidate_set,self.ref_cell_dic)
        return self.label_cluster



class AnnotationDataset(scRNADataset):
    def __init__(self, **args):
        super().__init__(**args)
        try:
            self.target_filename = args['target']
        except:
            raise Exception("Please specify target file name")
        self.dataset_names = ["ref","target"]
        self.current_resolution = 1
        self.all_cell_dic = None
        self.all_label_cluster = None
    def initialize(self,ct_keys=None,batch_correction=True,resolution=0,KNN=False,target_build='ONES',tree:Tree=None,masked_celltype=None):
        """
        Initialize the dataloader and generate partial label

        :param ct_keys: Key for cell type in target data
        :param batch_correction: Whether to perform batch correction
        :param resolution: Resolution for tree node merge
        :param KNN: Whether to use KNN for partial label generation
        :param target_build: Whether to use HC or ONES for target data
        :param tree: Tree for label cluster
        :param masked_celltype: Cell type to be masked
        :except Exception: Raise exception if data file is not found
        :return: partial_Y: partial label for training
                cell_dic: Cell type for each label
        """
        self.readData()
        self.current_resolution = resolution
        # Preprocessing of data
        self.all_data = ad.concat({'ref':self.ref_data, 'target':self.target_data}, axis=0,label='dataset')
        self.all_data = normalize(self.all_data,normalize_input=True)
        if batch_correction == 'scale':
            sc.pp.scale(self.all_data)
        elif batch_correction:
            sc.pp.combat(self.all_data, key='dataset')
        if masked_celltype != None:
            ref_index = self.all_data.obs['dataset'] == 'ref'
            masked_index = self.all_data.obs['cell_type'].apply(lambda x:x in masked_celltype)
            self.all_data = self.all_data[~(masked_index & ref_index)]
        ref_index = self.all_data.obs['dataset'] == 'ref'
        target_index = self.all_data.obs['dataset'] == 'target'
        ref_cell_dic = np.unique(self.all_data[ref_index].obs['cell_type'])
        target_cell_dic = np.unique(self.all_data[target_index].obs['cell_type'])
        if set(target_cell_dic).difference(set(ref_cell_dic)) != set():
            target_difference = set(target_cell_dic).difference(set(ref_cell_dic))
            self.all_cell_dic = np.concatenate([ref_cell_dic,list(target_difference)])
            self.all_label_cluster = torch.arange(len(ref_cell_dic),len(self.all_cell_dic))
        else:
            self.all_cell_dic = ref_cell_dic
            self.all_label_cluster = torch.Tensor([])
        self.ref_cell_dic = ref_cell_dic
        self.all_data.obs['cell_label'] = pd.Categorical(self.all_data.obs['cell_type'], categories=self.all_cell_dic).codes
        ref_pp_data = self.all_data[ref_index]
        tar_pp_data = self.all_data[target_index]
        self.label_cluster = torch.arange(len(self.ref_cell_dic))
        if ct_keys != None:
            assert ct_keys in self.target_data.obs.columns ,"key not found in target data"
            train_labels = pd.Categorical(self.target_data.obs[ct_keys], categories=self.ref_cell_dic).codes
            if KNN:
                target_partial_Y = KNNRefPartialLabels(tar_pp_data, train_labels, self.ref_cell_dic)
            else:
                self.HC_Tree_buildup(ref_pp_data)
                self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
                self.label_cluster = self.computeLabelCluster(self.candidate_set, self.ref_cell_dic)
                target_partial_Y,_ = self.generateCandidateLabels(train_labels, self.ref_cell_dic)
        else:
            if target_build == 'HC':
                if tree == None:
                    print("No reference tree provided, generating tree")
                    prototype = self.HC_Tree_buildup(ref_pp_data)
                else:
                    self.tree = tree
                self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
                self.label_cluster = self.computeLabelCluster(self.candidate_set, self.ref_cell_dic)
                target_partial_Y = np.ones((tar_pp_data.n_obs, len(self.ref_cell_dic)))
            else:
                target_partial_Y = np.ones((tar_pp_data.n_obs, len(self.ref_cell_dic)))
        ref_label = ref_pp_data.obs['cell_label']
        ref_partial_Y = self.generateRefLabels(ref_label, self.ref_cell_dic)
        self.partial_Y = np.concatenate([ref_partial_Y, target_partial_Y], axis=0)
        self.all_data.obsm['partial_label'] = self.partial_Y
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
    def generateRefLabels(self,train_labels,label_ref):
        K = int(len(label_ref))
        partialY= np.zeros((len(train_labels),K))
        for i in np.unique(train_labels):
            samples = np.where(train_labels == i)[0]
            partialY[samples, :] = np.repeat(np.eye(K)[i, :].reshape(1, K), len(samples), axis=0)
        return partialY
    def generateCandidateLabels(self, train_label, label_ref,partial_rate=0,resolution=0):
        """
        Base function for partial label generation

        :param train_label: Sample label
        :param partial_rate: Probability to generate a false positive within candidate set
        :param label_ref: Cell type corresponding to each label, just for
        :param resolution: Resolution for tree node merge
        :return: partial label and None(For compatibility with Atlas dataset)
        """
        active_candidateset = [subset for subset in self.candidate_set if len(subset) > 1]
        active_ct = [[node.tag for node in subset] for subset in active_candidateset]
        print(active_ct)
        return generate_const_partial_labels(train_labels=train_label,label_ref=label_ref,candidate_set=active_ct), None
    def assignLabels(self,prototypes,target_data):
        """
        Assign labels to target data based on prototypes

        :param prototypes:
        :param target_data:
        :return:
        """
        target_possible_labels = []
        for i in range(target_data.n_obs):
            each_obs = target_data.X[i]
            if isinstance(each_obs,scipy.sparse.csr.csr_matrix):
                each_obs = each_obs.todense()
            dist = np.linalg.norm(prototypes - each_obs, axis=1)
            target_possible_labels.append(np.argmin(dist))
        return target_possible_labels

    def reGenerateLabels(self,assign_target_label,batch_size=64):
        """
        Assign labels to target data based on prototypes, called only for train mode

        :param prototypes:
        :param target_data:
        :return:
        """
        target_partial_Y, _ = self.generateCandidateLabels(assign_target_label, self.ref_cell_dic)
        ref_label = self.all_data[self.all_data.obs['dataset'] == "ref"].obs['cell_label']
        ref_partial_Y = self.generateRefLabels(ref_label, self.ref_cell_dic)
        self.partial_Y = np.concatenate([ref_partial_Y, target_partial_Y], axis=0)
        self.all_data.obsm['partial_label'] = self.partial_Y
        return self.partial_Y, self.load(batch_size=batch_size)

    def computeLabelCluster(self,candidate_set,cell_dic):
        label_cluster = torch.zeros(len(cell_dic))
        for i in range(len(candidate_set)):
            cluster_labels = np.array([np.where(cell_dic == node.tag)[0][0] for node in candidate_set[i]])
            label_cluster[cluster_labels] = i
        print(label_cluster)
        return label_cluster

    def getLabelCluster(self,regenerate=False,all=False):
        if regenerate:
            self.current_resolution += 1
            if self.current_resolution > self.tree.depth():
                raise Exception("No more levels in the tree")
            self.candidate_set = createCandidateSet(self.tree, resolution=self.current_resolution)
            self.label_cluster = self.computeLabelCluster(self.candidate_set,self.ref_cell_dic)
        if all:
            return torch.concat([self.label_cluster,self.all_label_cluster])
        else:
            return self.label_cluster

    def checkCellTypeConsist(self,cell_dic,tree):
        ct_from_tree = [node.tag for node in tree.leaves()]
        cell_dic = list(cell_dic)
        if set(cell_dic).difference(set(ct_from_tree))!= set():
            print("Possible Novel Cell Detected")
        cell_dic += list(set(ct_from_tree).difference(set(cell_dic)))
        cell_dic = np.array(cell_dic)
        return cell_dic
    def getCellDic(self):
        return self.all_cell_dic
def generateTree(clusters,cell_dic)->Tree:
    tree = Tree()
    tree.create_node(identifier=-1, tag="ROOT", data=CellTypeNode(is_ct=True, type='ROOT'))
    for cluster in np.unique(clusters):
        cluster_elements = [i for i in range(len(cell_dic)) if clusters[i] == cluster]
        tree.create_node(identifier=cluster + len(cell_dic), tag="Cluster_{}".format(cluster), parent=-1,
                         data=CellTypeNode(is_ct=False, type="Cluster_{}".format(cluster)))
        for j in cluster_elements:
            tree.create_node(identifier=j, tag=cell_dic[j], parent=cluster + len(cell_dic),
                             data=CellTypeNode(is_ct=True, type=cell_dic[j]))
    tree.show(data_property='type')
    return tree
