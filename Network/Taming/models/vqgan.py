import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from Network.Taming.util import instantiate_from_config
from Network.Taming.modules.diffusionmodules.model import Encoder, Decoder
from Network.Taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from semivq import VectorQuant

class VQModel(pl.LightningModule):
    """ The model of VQVAE contains different methods. And in this repository, we do not 
        use perceptual loss and we only use MSE to train VQ-VAE model.
        params:
            ddconfig      -> dict: the parameters for backbone of model (resnet)
            lossconfig    -> dict: the codebook_weight is the weight for quantize loss 
            (loss_quantize = code_weight * ( (1-beta) * |z_e.detach() - z_q| + beta * |z_q.detach() - z_e|)
            vq_params     -> dict: parameters for VQ-VAE setting
            learning_rate -> int: learning rate for model
            ck_pt_path    -> str: the path of the model
        return:
    """
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 vqparams={},
                 learning_rate=1e-4,
                 ckpt_path=None,
                 ):
        super().__init__()
        self.n_embed = vqparams['num_codes']
        self.embed_dim = vqparams['feature_size']
        self.codebook_weight = lossconfig['codebook_weight']
        self.ignore_commitmentloss = False

        # For VQ-STE++, they split object function as two parts(Reconstuction part and codebook part)
        # This inplace_optimizer is prepared for codebook part optimization.
        if 'inplace_optimizer' in vqparams and vqparams['inplace_optimizer']:
            assert 'beta' in vqparams and vqparams['beta'] == 1
            inplace_optimizer = lambda *args, **kwargs: torch.optim.SGD(*args, **kwargs, lr=100, momentum=0.9)
            vqparams['inplace_optimizer'] = inplace_optimizer
            self.ignore_commitmentloss = True
        else:
            vqparams['inplace_optimizer'] = None

        # init the backbone of the model
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        
        # set lr training epoch and warm epoch
        self.learning_rate = learning_rate
        self.warm_iter = 0
        self.max_iter = 0
        
        # init quantize part
        self.quantize = VectorQuant(**vqparams)
        
        # make the input and output as wanted shape
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], self.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)


    def init_from_ckpt(self, path):
        """ init model from certain shape """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        self.load_state_dict(sd, strict=True)
        print(f"Restored from {path}")

    def encode(self, x):
        """ the forward to encode input"""
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, to_return = self.quantize(h)
        return quant, to_return

    def decode(self, quant):
        """ the forward of decoder"""
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        """ get quantized code and decode it"""
        quant_b = self.quantize.get_codebook_entry(code_b.view(-1), (-1, code_b.size(1), code_b.size(2), self.embed_dim))
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, to_return = self.encode(input)
        diff = to_return['loss']
        dec = self.decode(quant)
        perpelxtiy = to_return['perplexity']
        active_ratio = to_return['active_ratio']
        distance = to_return['d']
        return dec, diff, [perpelxtiy, active_ratio, distance]

    def get_input(self, batch):
        """ get input of dataset """
        x = batch[0]
        return x.float()

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, arr = self(x)
        perplexity, active_ratio, distance = arr[0], arr[1], torch.mean(arr[2])
        rec_loss = ((x.contiguous() - xrec.contiguous()) ** 2).mean()
        if self.ignore_commitmentloss:
            ae_loss = rec_loss
        else:
            ae_loss = rec_loss + self.codebook_weight * qloss
        log_dict_ae = {
            "train/active" : active_ratio,
            "train/perplexity" : perplexity,
            "distance" : distance,
            "train/recloss" : rec_loss,
            "train/cmtloss" : qloss,
            "train/aeloss"  : ae_loss,
        }
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return ae_loss


    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, arr = self(x)
        perplexity, active_ratio, distance = arr[0], arr[1], torch.mean(arr[2])
        rec_loss = ((x.contiguous() - xrec.contiguous()) ** 2).mean()
        if self.ignore_commitmentloss:
            ae_loss = rec_loss
        else:
            ae_loss = rec_loss + self.codebook_weight * qloss
        log_dict = {
            "val/active" : active_ratio,
            "val/perplexity" : perplexity,
            "val/distance" : distance,
            "val/recloss" : rec_loss,
            "val/cmtloss" : qloss,
            "val/aeloss"  : ae_loss,
        }
        self.log_dict(log_dict)
        return self.log_dict
    
    def test_step(self, batch, batch_idx):
        x = self.get_input(batch)
        xrec, qloss, arr = self(x)

        perplexity, active_ratio, distance = arr[0], arr[1], torch.mean(arr[2])
        rec_loss = ((x.contiguous() - xrec.contiguous()) ** 2).mean()
        ae_loss = rec_loss + self.codebook_weight * qloss
        log_dict = {
            "test/active" : active_ratio,
            "test/perplexity" : perplexity,
            "test/distance" : distance,
            "test/recloss" : rec_loss,
            "test/cmtloss" : qloss,
            "test/aeloss"  : ae_loss,
        }
        self.log_dict(log_dict)
        return self.log_dict


    def get_cosine_scheduler(self, optimizer, max_lr, min_lr, warmup_iter, base_lr, T_max):
        """ Get a cosine scheduler of learning rate for certain warm up epoch
            params:
                    optimizer   -> torch.optim
                    max_lr      -> double: the maximum of learning rate (the highest point of cosine curve)
                    min_lr      -> double: the minmum of learning rate (the lowest point of cosine curve)
                    warmup_iter -> int: the total iterations to use cosine scheduler
                    base_lr     -> int: base learning rate 
                    T_max       -> int: the total of training iterations
            return:
        """
        rule = lambda cur_iter : 1.0 if warmup_iter < cur_iter else (min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos((cur_iter - warmup_iter) / (T_max-warmup_iter) * math.pi))) / base_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule)
        return scheduler

    def configure_optimizers(self):
        """ get optimizer and lr scheduler """
        lr = self.learning_rate
        opt_ae = torch.optim.AdamW(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.9, 0.95),
                                  weight_decay=3e-4
                                  )
        sche_ae = self.get_cosine_scheduler(
                                            optimizer=opt_ae,
                                            max_lr=lr,
                                            min_lr=lr/10,
                                            warmup_iter=self.warm_iter,
                                            base_lr=lr,
                                            T_max=self.max_iter
                                            )
        return [opt_ae], [{"scheduler":sche_ae, "interval":"step"}]


    def set_iter_config(self, warm_iter, max_iter):
        self.warm_iter = warm_iter
        self.max_iter = max_iter

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def getcodebook(self):
        """get all codebook vectors"""
        return self.quantize.get_codebook()


