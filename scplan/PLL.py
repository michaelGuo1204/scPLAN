import lightning.pytorch as pl
import numpy as np
import pandas as pd
import anndata as ad
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateFinder
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

from scplan.Novel import computeEnsemble,mixup
from scplan.Critic import ClsLoss, SupConLoss, ZINBLoss,NeigborLoss
from scplan.Model import Model


class PartialLabelLearning(pl.LightningModule):
    """
    PLAN worker class
    """
    def __init__(self, param):
        """
        Initializing the worker by plan.Param

        :param param: User specific params
        """
        super().__init__()
        self.param = param
        self.model = None
        self.optimizer = None
        self.train_loader = None  # Training data loader
        self.part_mat = None      # Candidate label matrix
        self.confidence = None    # Confidence of candidate label matrix
        self.critic = {}          # Loss function dict
        self.train_metric = {}    # Training metric dict
        self.datasetofintrest = None # Dataset of interest
        self.final_level = False  # Whether the on the last leve of the tree
        self.confident_index = [] # Index of confident cells
        self.proto_start = False
        self.update_cluster = False
        self.hasnovel = False
        self.calm = False

    def Setup(self, encoder, pretrain=False, simu_param=None, logger=None, dataset=None):
        """
        Setup the worker by encoder type, whether to use pretrained plan, simulation params, logger and dataset.
        If dataset is not provided, simulation is used, else real data is used.
        Pretrain would be performed if pretrain is True. A pl.trainer instance is used for pretraining with early stopping
        and optimal learning rate finding.

        :param encoder: Encoder type, e.g. BasicEncoder or ZINBEncoder
        :param pretrain: Whether to pretrain encoder
        :param simu_param: If simulation is used, specify the simulation params
        :param logger: pl.logger instances for logging
        :param dataset: If real data is used, specify the dataset
        :return:
        """
        try:
            self.dataset = dataset                       #
            self.part_mat= (                                  # Get candidate label matrix
                self.dataset.getPartialLabel())
            self.train_loader = (                               # Get training data loader
                self.dataset.load(self.param.batch_size))
            self.part_mat = torch.Tensor(self.part_mat)         # Convert to torch.Tensor
            self.datasetofintrest = self.dataset.dataset_names[-1]
        except Exception as e:
            logger.error("{}".format(e.args))
            return

        tempY = self.part_mat.sum(dim=1).unsqueeze(1).repeat(1, self.part_mat.shape[1])
        candidate = self.part_mat.float() / tempY
        candidate = candidate.to(self.param.device[0])
        self.critic["Cls"] = ClsLoss(candidate, novel_cell=self.param.novel_cell)
        self.critic["Cls"].updateLabelClusters(self.dataset.getLabelCluster())
        self.critic["Con"] = SupConLoss()
        self.critic["ZINB"] = ZINBLoss()
        self.critic["Neighbor"] = NeigborLoss()
        pretrain_model = encoder(num_class=self.param.num_class, encoder_dim=self.param.enc_dim,
                                 data_dim=self.param.num_features, latent_dim=self.param.latent_dim,
                                 decoder_dim=self.param.dec_dim,novel_cell=self.param.novel_cell)# Initialize encoder
        if pretrain:
            trainer = pl.Trainer(max_epochs=self.param.pretrain_epoch, logger=logger, devices=self.param.device,
                                 callbacks=[LearningRateFinder(num_training_steps=100, max_lr=self.param.lr * 10,min_lr=self.param.lr / 10),
                                            EarlyStopping(monitor="Pretrain_loss", mode="min", min_delta=0.0001,patience=20)])
            trainer.fit(pretrain_model, self.train_loader)
            if pretrain_model.learning_rate != self.param.lr:
                self.param.lr = pretrain_model.learning_rate
        try:
            self.model = Model(self.param, encoder, pretrain=pretrain_model)
        except Exception as e:
            logger.error("{}".format(e))
            return
    def setup(self,stage):
        """
        Initialization procedure of pl.LightningModule, here we only perform
        hyperparameter logging before training stage

        :param stage: pl.LightningModule stage, e.g. fit or predict
        :return: None
        """
        if stage == "fit":
            param_dic = {"lr": self.param.lr, "batch_size": self.param.batch_size, "part": self.param.partial_rate,
                         "dec_dim": self.param.dec_dim, "enc_dim": self.param.enc_dim,
                         "latent_dim": self.param.latent_dim,"recon_loss_weight":self.param.loss_weight}
            self.logger.log_hyperparams(param_dic)
    def configure_optimizers(self):
        """
        Configure optimizer for PLAN

        :return: None
        """
        return torch.optim.Adam(self.model.parameters(), self.param.lr, weight_decay=self.param.weight_decay)

    def training_step(self,batch,batch_index):
        """
        Main training step of PLAN

        :param batch: batch data
        :param batch_index: batch index
        :return: Overall loss
        """
        X,X_drop,X_raw,sf,Y,Y_true,dataset,index = batch              # Get batch data
        if len(index) != self.param.batch_size: return                # Skip the last batch if it is not full
        cluster_Y = self.dataset.getLabelCluster().to(X)              # Get label cluster
        dataset = np.array(dataset)                                   # Convert dataset to numpy array
        ret = self.model(in_q=X, in_k=X_drop, partial_Y=Y,cluster_Y=cluster_Y, dataset=dataset) # Forward
        cls_out,proto_cls, latent, latent_label, proto_assign,all_datasets, encoder_out = ret # Unpack outputs
        latent_label = latent_label.contiguous().view(-1, 1)          # Reshape latent label
        loss_cls = 0
        if self.proto_start:               # Whether updating candidate label matrix
            self.critic["Cls"].confidence_update(proto_assign=proto_assign, index=index, candidate_label=Y)
                                                                      # Update confidence distribution of candidate label matrix
            mask = torch.eq(latent_label[:self.param.batch_size], latent_label.T).float().cuda()
                                                                      # Select samples of same class
            loss_cls += self.critic["Cls"].proto_forward(proto_cls)
            if self.param.ablation:                                   # If contrasive loss ablation is used
                all_datasets = np.array(all_datasets)
                all_datasets = np.expand_dims(all_datasets, axis=1)
                dataset_mask = np.equal(all_datasets[:self.param.batch_size], all_datasets.T)
                                                                      # Only samples from same dataset with same labels are selected
                dataset_mask = torch.Tensor(dataset_mask).float().to(latent)
                mask = dataset_mask * mask
        else:
            mask = None
        if self.hasnovel:                                            # If novel cell is detected
            loss_neighbor = self.critic["Neighbor"](latent=latent[:self.param.batch_size],proto_assign=proto_assign,cls_out=cls_out)
        else: loss_neighbor = 0
        loss_recons = self.critic["ZINB"](x=X_raw, scale_factor=sf, **encoder_out)
        loss_cont = self.critic["Con"](features=latent, mask=mask, batch_size=self.param.batch_size)
        loss_cls += self.critic["Cls"](cls_out, index)                # Compute contrastive and classification loss
        loss = ( loss_cls + self.param.loss_weight * loss_cont) + loss_recons * self.param.recon_weight + 0.5*loss_neighbor
        with torch.no_grad():                                         # Compute training metric
            cls_oi = cls_out[dataset == self.datasetofintrest]        # Compute accuracy of dataset of interest
            proto_oi = proto_assign[dataset == self.datasetofintrest]
            Y_true_oi = Y_true[dataset == self.datasetofintrest]
            cluster_all = self.dataset.getLabelCluster(all=True).to(X)
            cluster_oi = cluster_all[Y_true_oi]
            _,cls_pseudo = torch.max(cls_oi,dim=1)
            _,proto_pseudo = torch.max(proto_oi,dim=1)
            cls_pred = cluster_all[cls_pseudo]
            proto_pred = cluster_all[proto_pseudo]
            self.train_metric['acc_cls'] += torch.where(cls_pred == cluster_oi,1,0).sum().item()
            self.train_metric['acc_proto'] += torch.where(proto_pred == cluster_oi,1,0).sum().item()
            self.train_metric['loss_value_cont'] += loss_cont.item()
            self.train_metric['loss_value_cls'] += loss_cls.item()
            self.train_metric['samples'] += len(Y_true_oi)
        return loss

    def on_train_epoch_start(self):
        """
        At the beginning of each training epoch, reset training metric
        """
        self.train_metric["samples"] = 0
        self.train_metric["acc_cls"] = 0
        self.train_metric["acc_proto"] = 0
        self.train_metric["loss_value_cls"] = 0
        self.train_metric["loss_value_cont"] = 0
        if self.param.debatch:
            self.train_metric["loss_align"] = 0
        if self.current_epoch >= self.param.epochs[0] and not self.calm:
            self.proto_start = True
        if self.current_epoch == self.param.epochs[1]:
            print("Updating label clusters")
            try:
                new_label_cluster = self.dataset.getLabelCluster(regenerate=True)
            except Exception:
                self.final_level = True
                print("Label cluster generation failed");return
            self.critic["Cls"].updateLabelClusters(new_label_cluster)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Dataloader overload for novel cell perception

        :return dataloader: dataloader

        """
        if self.param.novel_cell and self.current_epoch in self.param.novel_epoch:
            print("\nChecking Novel Cells")
            self.model.eval()
            hasnovel = self.validateNovelCell()
            self.model.train()
            if hasnovel:
                self.hasnovel = True
                threshold, masked_index = hasnovel
                self.confident_index = masked_index
                print("Novel cell detected, updating dataloaders, current threshold:{}, confident target cells:{}".format(threshold,len(masked_index)))
                return self.dataset.reloadloader(batch_size=self.param.batch_size,mask=np.array(masked_index))
        return self.train_loader

    def on_train_epoch_end(self):
        """
        At the end of each training epoch, log training metric
        """
        self.train_metric['acc_cls'] /= self.train_metric['samples']
        self.train_metric['acc_proto'] /= self.train_metric['samples']
        self.critic["Cls"].momentum_schedule(self.current_epoch, self.param)
        mmc = self.critic["Cls"].candidate.max(dim=1)[0].mean()
        self.train_metric["mmc"] = mmc
        self.logger.log_metrics(self.train_metric,step=self.current_epoch)
        self.hasnovel = False
        self.calm = False
        self.proto_start = False

    def predict_step(self, batch, batch_idx):
        """
        Prediction step of PLAN, store raw data, cell type labels, encoder latents and classifier predictions

        :param batch: batch data
        :param batch_idx: batch index
        :return: None
        """
        X,X_dropout,X_raw,sf,Y,Y_true,dataset,index = batch
        cls_out,latent,_ = self.model(in_q=X,partial_Y=Y)
        return {'raw': X_raw, 'cell_type': Y_true, 'latent': latent, 'pred': cls_out,'index':index,'dataset':dataset}

    def getPredictions(self,pred_outputs):
        keys = ['raw','cell_type','latent','pred','index']
        final_data = {key:[] for key in keys+['dataset']}
        for batch in pred_outputs:
            for key in keys:
                final_data[key].append(batch[key].cpu().numpy())
            final_data['dataset'].append(batch['dataset'])
        final_data = {key:np.concatenate(item) for key,item in final_data.items() }
        var_df ,_ = self.dataset.getRawDataMeta()
        cell_dic = self.dataset.getCellDic()
        final_adata = ad.AnnData(final_data['raw'],var=var_df)
        final_adata.obs['cell_type'] = pd.Categorical(final_data['cell_type'],categories=np.arange(len(cell_dic))).rename_categories(cell_dic)
        final_adata.obs['pred'] = pd.Categorical(final_data['pred'],categories=np.arange(len(cell_dic))).rename_categories(cell_dic)
        final_adata.obs['dataset'] = final_data['dataset']
        final_adata.obsm['latent'] = final_data['latent']
        if self.param.novel_cell:
            novel_pred = pd.Series(["Unknown" for _ in range(final_adata.n_obs)])
            confident_index = set(self.confident_index) | set(np.where(final_data['dataset']=='ref')[0])
            confident_index = np.array(list(confident_index))
            novel_pred[confident_index] = final_adata.obs['pred'][confident_index]
            self.model.eval()
            self.model.to(self.param.device[0])
            entropy,softmax,confidence,ensemble_index = self.validateNovelCell(pred=True)
            self.model.to('cpu')
            entropy_pred = np.array([np.inf for _ in range(final_adata.n_obs)])
            entropy_pred[ensemble_index.int().numpy()] = entropy.numpy()
            softmax_pred = np.array([np.inf for _ in range(final_adata.n_obs)])
            softmax_pred[ensemble_index.int().numpy()] = softmax.numpy()
            confidence_pred = np.array([np.inf for _ in range(final_adata.n_obs)])
            confidence_pred[ensemble_index.int().numpy()] = confidence.numpy()
            final_adata.obs['pred_with_novel'] = pd.Categorical(novel_pred)
            final_adata.obs['entropy'] = entropy_pred
            final_adata.obs['softmax'] = softmax_pred
            final_adata.obs['confidence'] = confidence_pred
        final_adata.obs_names = self.dataset.getDataIndex()[final_data['index']]
        return final_adata


    def saveModel(self,path):
        """
        Save plan to path

        :param path: string path
        """
        torch.save(self.model.state_dict(),path)
    def loadModel(self,path,encoder):
        """
        Load plan from path and set to eval mode

        :param path: Model path
        :param encoder: Model encoder type, shall be consist with the plan to be loaded
        """
        self.model = Model(self.param,encoder,pretrain=None)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def evaAcc(self,output,labels,topk=3):
        topk = min(topk,self.param.num_class)
        with torch.no_grad():
            _, pred = output.topk(topk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(labels.view(1, -1).expand_as(pred))
            res = []
            for k in range(1,topk):
                correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
                res.append(correct_k.cpu().numpy()[0])
            res = np.array(res)
            return res

    @torch.no_grad()
    def validateNovelCell(self,debug=True,pred=False):
        all_latent = torch.Tensor()
        cls_out_all = torch.Tensor()
        proto_confidence_all = torch.Tensor()
        data_index = torch.Tensor()
        cluster_Y = self.dataset.getLabelCluster().to(self.param.device[0])              # Get label cluster
        cluster_label = torch.unique(cluster_Y)
        #mask = torch.where(cluster_Y.reshape(-1, 1) == cluster_label, 1, 0).to(self.param.device[0])
        mask = torch.where(cluster_label == cluster_Y.reshape(-1, 1), 1, 0).float().to(self.param.device[0])
        #mask = torch.eye(cluster_Y.shape[0]).to(self.param.device[0])
        #mask = mask / mask.sum(dim=0)
        dataloader = self.dataset.load(batch_size=self.param.batch_size, train=False) if pred \
            else self.train_loader
        for batch_idx, batch in enumerate(dataloader):
            X, X_drop, X_raw, sf, Y, Y_true, dataset, index = batch
            X_cuda = X.to(self.param.device[0])
            Y_cuda = Y.to(self.param.device[0])
            pseudo_label,latent,cls_out_naive= self.model(in_q=X_cuda, partial_Y=Y_cuda)
            prototype = self.model.prototypes.clone().detach()
            similarity = torch.mm(latent, prototype.t())
            proto_cls_confidence,proto_psedu_label = torch.max(similarity,dim=1)
            #proto_cls_confidence = (proto_cls_confidence - proto_cls_confidence.min())/(proto_cls_confidence.max()-proto_cls_confidence.min())
            #cls_out, _ = (mask.unsqueeze(0) * cls_out_naive.unsqueeze(2)).max(dim=1)
            cls_out = cls_out_naive @ mask
            target_index = np.where(np.array(dataset) == self.datasetofintrest)[0]
            proto_confidence_all = torch.cat([proto_confidence_all,proto_cls_confidence.cpu()[target_index]],dim=0)
            all_latent = torch.cat([all_latent,latent.cpu()[target_index]],dim=0)
            cls_out_all = torch.cat([cls_out_all,cls_out.cpu()[target_index]],dim=0)
            data_index = torch.cat([data_index,index[target_index]])
        novel_exist,entropy,softmax = computeEnsemble(cls_out_all,proto_confidence_all)
        ensemble_score = (entropy + softmax)/2
        if pred: return entropy,softmax,proto_confidence_all,data_index
        if debug: histEnsemble(ensemble_score)
        if novel_exist:
            if all_latent.shape[0] > 1000:
                idx_rand = torch.randperm(all_latent.shape[0])[:1000]
                all_latent_partial = all_latent[idx_rand]
            else: all_latent_partial = all_latent
            latent_mix = mixup(all_latent_partial)
            latent_norm = torch.nn.functional.normalize(latent_mix,dim=1,p=2).to(self.param.device[0])
            main_mix_out = self.model.encoder_q.clsforward(latent_norm)
            main_mix_out = main_mix_out @ mask
            prototype = self.model.prototypes.clone().detach()
            mixup_similarity = torch.mm(latent_norm, prototype.t())
            mixup_confidence,_ = torch.max(mixup_similarity,dim=1)
            #mixup_confidence = (mixup_confidence - mixup_confidence.min())/(mixup_confidence.max()-mixup_confidence.min())
            #main_mix_out, _ = (mask.unsqueeze(0) * main_mix_out.unsqueeze(2)).max(dim=1)
            mixup_entropy,mixup_softmax = computeEnsemble(main_mix_out,mixup_confidence,score_only=True)
            mixup_ensemble = (mixup_softmax+mixup_entropy)/2
            mixup_ensemble = mixup_ensemble.cpu().numpy()
            threshold = np.quantile(mixup_ensemble,0.1)
            ensemble_mask = torch.where(ensemble_score > threshold)[0]
            masked_index = data_index[ensemble_mask].long()
        ret = (threshold,masked_index) if novel_exist else False
        return ret

def histEnsemble(ensemble):
    import matplotlib.pyplot as plt
    plt.hist(ensemble.cpu().numpy(),bins=20)
    plt.show()
