import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from scplan.Critic import ZINBLoss, dispDec, muDec


class ZINBEncoder(pl.LightningModule):
    """
    ZINB AutoEncoder for PLAN
    """

    def __init__(self, latent_dim, num_class, encoder_dim, decoder_dim, data_dim, pretrain=None, novel_cell=False):
        """
        Initializing the encoder

        :param latent_dim: Latent dimension
        :param num_class: Number of classes, specifying the dimension of the classifier output
        :param encoder_dim: Hidden encoder dimension
        :param decoder_dim: Hidden decoder dimension
        :param data_dim: Input data dimension
        :param pretrain: Whether to use pretrained plan
        """
        super().__init__()
        self.AE = ZINBAutoEncoder(
            data_dim=data_dim,
            num_class=num_class,
            encoder_dim=encoder_dim,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            novel_cell=novel_cell,
        )
        if pretrain != None:
            self.AE.load_state_dict(pretrain.AE.state_dict())
        self.zloss = ZINBLoss()
        self.learning_rate = 1e-3

    def configure_optimizers(self):
        return Adam(self.AE.parameters(), lr=self.learning_rate, amsgrad=True)

    def training_step(self, batch, batch_index):
        X, X_dropout, X_raw, sf, part_label, true_label, dataset, _ = batch
        _, _, encoder_pack = self.forward(X)
        loss = self.zloss(x=X_raw, scale_factor=sf, **encoder_pack)
        self.log("Pretrain_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def forward(self, x):
        return self.AE(x)

    def clsforward(self, x):
        return self.AE.clsforward(x)

    def auxclsforward(self, x):
        return self.AE.auxclsforward(x)


class ZINBAutoEncoder(nn.Module):
    """
    ZINB AutoEncoder
    """

    def __init__(self, data_dim, num_class, encoder_dim, latent_dim, decoder_dim, novel_cell=False):
        super().__init__()
        self.sigma = 0.1
        self.encoder = buildNetwork([data_dim] + encoder_dim + [latent_dim], activation="relu")
        self.decoder = buildNetwork([latent_dim] + decoder_dim)
        self.classifier = nn.Linear(latent_dim, num_class)
        self.mu_decoder = nn.Sequential(nn.Linear(decoder_dim[-1], data_dim), muDec())
        self.disp_decoder = nn.Sequential(nn.Linear(decoder_dim[-1], data_dim), dispDec())
        self.pi_decoder = nn.Sequential(nn.Linear(decoder_dim[-1], data_dim), nn.Sigmoid())
        self.novel_cell = novel_cell

    def forward(self, x):
        """
        Forward pass of ZINB AutoEncoder

        :param x: input data
        :return: Classifier Predictions,Latent embeddings
        """
        z = self.encoder(x + torch.randn_like(x) * self.sigma)
        d = self.decoder(z)
        _mean = self.mu_decoder(d)
        _disp = self.disp_decoder(d)
        _pi = self.pi_decoder(d)
        feat = self.encoder(x)
        feat_norm = F.normalize(feat, p=2, dim=1)
        logits = self.classifier(feat_norm)
        return logits, feat_norm, {"mean": _mean, "disp": _disp, "pi": _pi}

    def clsforward(self, z):
        return self.classifier(z)


def buildNetwork(layers, activation="relu", layer_norm=False):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if i != len(layers) - 1:  # TODO: Check correctness
            if activation == "relu":
                net.append(nn.ReLU())
            elif activation == "sigmoid":
                net.append(nn.Sigmoid())
    if layer_norm:
        net.append(nn.LayerNorm(layers[-1]))
    return nn.Sequential(*net)
