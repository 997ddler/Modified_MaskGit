import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from Network.Taming.util import instantiate_from_config
from Network.Taming.modules.diffusionmodules.model import Encoder, Decoder
from Network.Taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from semivq import VectorQuant

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 vqparams={},
                 learning_rate=1e-4,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key 
        # self.n_embed = vqparams['num_codes']

        # self.embed_dim = vqparams['feature_size']
        self.codebook_weight = lossconfig['codebook_weight']

        self.n_embed = n_embed
        self.embed_dim = embed_dim

        if 'inplace_optimizer' in vqparams:
            assert 'beta' in vqparams and vqparams['beta'] == 1
            inplace_optimizer = lambda *args, **kwargs: torch.optim.AdamW(
                                                                          *args,
                                                                          **kwargs,
                                                                          lr=learning_rate,
                                                                          weight_decay=1.0e-4,
                                                                          betas=(0.9, 0.95)
                                                                          ) 
            vqparams['inplace_optimizer'] = inplace_optimizer
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.learning_rate = learning_rate
        # self.loss = instantiate_from_config(lossconfig)
        self.warm_iter = 0
        self.max_iter = 0
        self.quantize = VectorQuantizer(self.n_embed, self.embed_dim, beta=0.25,
                                         remap=remap, sane_index_shape=sane_index_shape)
        #self.quantize = VectorQuant(**vqparams)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info
        # quant, to_return = self.quantize(h)
        #print('shape of quant:' + str(quant.shape))
        #print('shape of index of quant:' + str(to_return['q'].shape))
        #return quant, to_return

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        #get codebook dimensions
        #print(self.quantize.e_dim, "e_dim")
        quant_b = self.quantize.get_codebook_entry(code_b.view(-1), (-1, code_b.size(1), code_b.size(2), self.embed_dim))
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, 0.0
        #quant, to_return = self.encode(input)
        #diff = to_return['loss']
        #dec = self.decode(quant)
        #perpelxtiy = to_return['perplexity']
        #return dec, diff, perpelxtiy

    def get_input(self, batch, k):
        x = batch[0]
        # if len(x.shape) == 3:
        #     x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, perplexity = self(x)

        rec_loss = ((x.contiguous() - xrec.contiguous()) ** 2).mean()
        ae_loss = rec_loss + self.codebook_weight * qloss.mean()
        log_dict_ae = {
            "train/perplexity" : perplexity,
            "train/aeloss"  : ae_loss,
            "train/recloss" : rec_loss,
            "train/cmtloss" : qloss,
        }
        # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return ae_loss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, perplexity = self(x)

        # rec_loss = torch.abs(x.contiguous() - xrec.contiguous()).mean()
        rec_loss = ((x.contiguous() - xrec.contiguous()) ** 2).mean()
        ae_loss = rec_loss + self.codebook_weight * qloss.mean()
        log_dict = {
            "val/aeloss"  : ae_loss,
            "val/recloss" : rec_loss,
            "val/perplexity" : perplexity,
            "val/cmtloss" : qloss,
        }
        self.log_dict(log_dict)
        # aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
        #                                    last_layer=self.get_last_layer(), split="val")

        #discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
        #                                    last_layer=self.get_last_layer(), split="val")
        #rec_loss = log_dict_ae["val/rec_loss"]
        #self.log("val/rec_loss", rec_loss,
        #           prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log("val/aeloss", aeloss,
        #           prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log("performance", {"val/rec_loss": rec_loss, "val/ae_loss": aeloss}, sync_dist=True)
        #self.log_dict(log_dict_ae)
        #self.log_dict(log_dict_disc)
        return self.log_dict

    def get_cosine_scheduler(self, optimizer, max_lr, min_lr, warmup_iter, base_lr, T_max):
        rule = lambda cur_iter : 1.0 if warmup_iter < cur_iter else (min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos((cur_iter - warmup_iter) / (T_max-warmup_iter) * math.pi))) / base_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
        return scheduler

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.9, 0.95),
                                  weight_decay=1e-4
                                  )
        sche_ae = self.get_cosine_scheduler(
                                            optimizer=opt_ae,
                                            max_lr=lr,
                                            min_lr=lr/10,
                                            warmup_iter=self.warm_iter,
                                            base_lr=lr,
                                            T_max=self.max_iter
                                            )
#        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                    lr=lr, betas=(0.9, 0.95),
#                                    weight_decay=1e-4
#                                    )
#        sche_disc = self.get_cosine_scheduler(
#                                              optimizer=opt_disc,
#                                              max_lr=lr,
#                                              min_lr=lr/10,
#                                              warmup_iter=self.warm_iter,
#                                              base_lr=lr,
#                                              T_max=self.max_iter
#                                              )
#        return [opt_ae, opt_disc], [{"scheduler":sche_ae, "interval":"step"}, {"scheduler":sche_disc, "interval":"step"}]
        return [opt_ae], [{"scheduler":sche_ae, "interval":"step"}]


    def set_iter_config(self, warm_iter, max_iter):
        self.warm_iter = warm_iter
        self.max_iter = max_iter

    def get_last_layer(self):
        return self.decoder.conv_out.weight


