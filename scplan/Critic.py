import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsLoss(torch.nn.Module):
    """
    Loss function for classifier training
    """
    def __init__(self, candidate, candidate_momentum=0.99,novel_cell=False):
        """
        Loss function initialization

        :param candidate: Candidate set of labels stored in loss, stored as torch bufer
        :param candidate_momentum: Momentum of updating candidate set of labels
        """
        super().__init__()
        self.register_buffer("candidate", candidate)
        self.candidate_momentum = candidate_momentum
        self.label_mask = None
        self.novel_cell = novel_cell
    def momentum_schedule(self, epoch, param):
        """
        Momentum schedule for updating candidate set of labels

        :param epoch: Current epoch
        :param param: Main param set of PLL
        """
        start = param.momentum_schedule[0]
        end = param.momentum_schedule[1]
        self.candidate_momentum = 1. * epoch / param.epochs[-1] * (end - start) + start

    def forward(self, prediction, index):
        """
        Loss computation based on classifier

        :param prediction: Softmax output of classfier prediction
        :param index: Index of batch in original data
        :param mask: Mask of source dataset
        """
        logtrans_pred = F.log_softmax(prediction, dim=1)
        masked_pred = logtrans_pred * self.candidate[index, :]
        average_loss = - ((masked_pred).sum(dim=1)).mean()
        return average_loss
    def updateLabelClusters(self, label_clusters):
        dim = len(label_clusters)
        self.label_mask = torch.where(label_clusters.repeat(dim, 1) == label_clusters.repeat(dim, 1).T, 1, 0).to(self.candidate)
    def proto_forward(self, prediction):
        """
        Loss computation based on prototype assignment

        :param prediction: Softmax output of prototype assignment
        :param index: Index of batch in original data
        """
        logtrans_pred = F.log_softmax(prediction, dim=1)
        masked_pred = logtrans_pred * self.label_mask / self.label_mask.sum(dim=1)
        average_loss = - ((masked_pred).sum(dim=1)).mean()
        return average_loss

    def confidence_update(self, proto_assign, index, candidate_label):
        """
        Update candidate set of labels by pseudo label prediction

        :param proto_assign: Protype assignment results of the plan
        :param index: Index of batch in original data
        :param candidate_label: Artificial candidate set of labels
        """
        with torch.no_grad():
            _, prot_pred = (proto_assign * candidate_label).max(dim=1)
            #pseudo_label = F.one_hot(prot_pred, candidate_label.shape[1]).float().cuda().detach()
            pseudo_label = self.label_mask[prot_pred] #/ self.label_mask[prot_pred].sum(dim=1,keepdims=True)
            self.candidate[index, :] = self.candidate_momentum * self.candidate[index, :] \
                                              + (1 - self.candidate_momentum) * pseudo_label
        return None


# 对比损失
class SupConLoss(nn.Module):
    """
    Supervised/Unsupervised Contrastive loss
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Loss function initialization

        :param temperature: Temperature used in contrastive loss
        :param base_temperature: Base temperature used in contrastive loss
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, mask=None, batch_size=-1):
        """
        Contrastive loss computation, if mask is provided, then it is supervised contrastive(SupCon) loss, otherwise is
        unsupervised contrastive(MoCo) loss

        :param features: Batch latent features
        :param mask: Selection in supervised contrastive loss for latent of same class
        :param batch_size: batch size of current batch
        :return:
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss

class NeigborLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmp = 0.1

    def forward(self, latent,cls_out,proto_assign):
        proto_assign = proto_assign.clone().detach()
        feat_mat = torch.mm(latent,latent.t()) / self.tmp
        #mask = torch.eye(feat_mat.size(0), feat_mat.size(0)).to(latent).bool()
        #feat_mat = torch.masked_fill(feat_mat, mask, -1/self.tmp)
        local_nb_dist, local_nb_idx = torch.max(feat_mat, 1)
        local_nb_cls = cls_out[local_nb_idx, :]
        local_nb_proto = proto_assign[local_nb_idx, :]
        local_loss = -torch.sum(proto_assign * F.log_softmax(local_nb_cls,dim=1))
        local_loss += -torch.sum(local_nb_proto * F.log_softmax(cls_out,dim=1))
        local_loss /= 2 * len(latent)
        return local_loss
class OTLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        pass


class ZINBLoss(pl.LightningModule):
    """
    Reconstruction loss in ZINB encoder
    """
    def __init__(self):
        super().__init__()
    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        """
        Reconstruction loss computation

        :param x: Original count matrix
        :param mean: Mean output of ZINB encoder
        :param disp: Dispersion output of ZINB encoder
        :param pi: Dropout rate output of ZINB encoder
        :param scale_factor: Size factor of original count matrix
        :param ridge_lambda: Lambda in ridge regression
        :return:
        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result


class muDec(nn.Module):
    def __init__(self):
        super(muDec, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class dispDec(nn.Module):
    def __init__(self):
        super(dispDec, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)
