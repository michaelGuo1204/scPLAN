import torch
import numpy as np
from lightning.pytorch import LightningModule


class Model(LightningModule):
    """
    Main PLAN model
    """

    def __init__(self, param, encoder, pretrain=None):
        """
        Initializing the plan

        :param param: Main PLAN params
        :param encoder: Encoder type
        :param pretrain: Whether using pretrained plan
        """
        super().__init__()
        self.param = param
        self.encoder_q = encoder(num_class=self.param.num_class, encoder_dim=self.param.enc_dim,
                                 data_dim=self.param.num_features, latent_dim=self.param.latent_dim,
                                 decoder_dim=self.param.dec_dim, pretrain=pretrain,novel_cell=self.param.novel_cell)
        self.encoder_k = encoder(num_class=self.param.num_class, encoder_dim=self.param.enc_dim,
                                 data_dim=self.param.num_features, latent_dim=self.param.latent_dim,
                                 decoder_dim=self.param.dec_dim, pretrain=None,novel_cell=self.param.novel_cell)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.register_buffer("queue", torch.randn(param.moco_queue, param.latent_dim))
        self.register_buffer("queue_pseudo", torch.randn(param.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("prototypes", torch.zeros(param.num_class, param.latent_dim))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.datasets = np.array(["target" for i in range(param.moco_queue)])

    @torch.no_grad()
    def encoderKUpdate(self):
        """
        Update key encoder parameters by query encoder in momentum scheme

        :return: None
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.param.moco_m + param_q.data * (1. - self.param.moco_m)

    @torch.no_grad()
    def queueUpdate(self, key, label, dataset):
        """
        Update queue by new output latents/labels

        :param key: Latent output of key encoder
        :param label: Pseudo-type assignments of key latent
        :param dataset: Dataset source of latent
        :return: None
        """
        batch_size = key.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr + batch_size, :] = key
        self.queue_pseudo[ptr:ptr + batch_size] = label
        self.datasets[ptr:ptr + batch_size] = dataset
        ptr = (ptr + batch_size) % self.param.moco_queue  # 移除指针
        self.queue_ptr[0] = ptr

    def forward(self, in_q, in_k=None, partial_Y=None, cluster_Y=None, dataset=None):
        """
        Forward pass of PLAN, two additional patterns are supported:
        1. Prediction: in_k is None, return latent and pseudo labels
        2. Batch removal: class_mask is not None and param.debatch specified, would calculate batch removal loss

        :param in_q: Input weak augmented data
        :param in_k: Input strong augmented data
        :param partial_Y: Candidate label of the data
        :param class_mask: Intra-dataset label mask in batch removal
        :param dataset: Dataset source of the batch
        :return:
        """
        q_cls, q_latent, encoder_out = self.encoder_q(in_q)  # Get query latent
        predicted_scores = torch.softmax(q_cls, dim=1) * partial_Y
        _, pseudo_labels_q = (
            torch.max(predicted_scores, dim=1))  # Get pseudo labels
        if not self.training:  # Return latent and pseudo labels if prediction
            return pseudo_labels_q, q_latent,q_cls
        prototype = self.prototypes.clone().detach()                    # Store last prototypes
        similarity = torch.mm(q_latent, prototype.t())                  # Calculate similarity
        proto_assign = torch.softmax(similarity, dim=1)                 # Calculate the most fit prototypes
        cluster_cls_assign = cluster_Y[pseudo_labels_q]                 # Get cluster assignments
        for latent, label in zip(q_latent, cluster_cls_assign):
            target_label = torch.where(cluster_Y == label)[0]           # Select target prototypes
            prototype[target_label] = (prototype[target_label] * self.param.proto_m  # Update prototypes
                                       + (1 - self.param.proto_m) * latent)
        self.prototypes = (
            torch.nn.functional.normalize(prototype, p=2, dim=1))  # Prototypes shall be normalized
        prototype_cls = self.encoder_q.clsforward(self.prototypes)  # Calculate prototype classifier
        with torch.no_grad():
            self.encoderKUpdate()  # Update key encoder by query encoder
            k_cls, k_latent, encoder_k_out = self.encoder_k(in_k)
        features = torch.cat((q_latent, k_latent,
                              self.queue.clone().detach()), dim=0)  # Combine all latents/labels
        pseudo_labels = torch.cat((cluster_cls_assign, cluster_cls_assign, self.queue_pseudo.clone().detach()), dim=0)
        self.queueUpdate(k_latent, cluster_cls_assign, dataset)  # Push key latent into queue
        all_datasets = np.concatenate([dataset,dataset, self.datasets],axis=None)
        return q_cls, prototype_cls, features, pseudo_labels, proto_assign, all_datasets, encoder_out

