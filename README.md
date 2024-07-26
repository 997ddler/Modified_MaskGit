# Modified MaskGIT

In this repository，we modify some part of MaskGIT (Original repository can  be found [here](https://github.com/valeoai/Maskgit-pytorch)). Specifically, We use [this repository](https://github.com/997ddler/Soft-discretization) to substitute original VQ-VAE. In our repo, we implement Least Recently Use Replace Policy, L2 normalization and VQ-STE++ etc to address the problem of index collapse of VQVAE.



## Repository Structure

```
  ├ MaskGIT-pytorch/
  |    ├── Metrics/                               <- evaluation tool
  |    |      ├── inception_metrics.py                  
  |    |      └── sample_and_eval.py
  |    |    
  |    ├── Network/                             
  |    |      ├── Taming/                         <- VQGAN architecture   
  |    |      └── transformer.py                  <- Transformer architecture  
  |    |
  |    ├── Trainer/                               <- Main class for training
  |    |      ├── trainer.py                      <- Abstract trainer     
  |    |      └── vit.py                          <- Trainer of maskgit
  |    ├── semivq                              	  <- Our repo implemented some Vector quantized method       
  |    |
  |    ├── requirements.yaml                      <- help to install env 
  |    ├── README.md                              <- Me :) 
  |    └── main.py                                <- Main
```





## Usage

### VQ-VAE

This is a config demo for VQ-VAE.

```
model:
  save_path: /home/zwh/Modified_MaskGit/pretrained_maskgit/VQGAN_Cifar_Semi # path to save your VQ-VAE model 
  learning_rate: 1.0e-4                                                     # learning rate for VQ
  target: vqgan.models.vqgan.VQModel										# the type of model loaded
  params:																
    ddconfig:																# the config for VQ backbone
      double_z: False
      z_channels: 64 # 256
      resolution: 64 
      in_channels: 3
      out_ch: 3
      ch: 32 # 128
      ch_mult: [ 1,1,2,2,4]  # num_down = len(ch_mult)-1
      num_res_blocks: 2
      attn_resolutions: [16]
      dropout: 0.0

    lossconfig:       														# quantize weight
      codebook_weight: 5.0


  vq_params:																# VQ configurations 
        num_codes: 1024
        feature_size: 64												 
        sync_nu: 2.0,										# synchronized update weight (VQ-STE++)
        inplace_optimizer: True								# alternated method （VQ-STE++)
        affine_lr: 2.0										# affine learning rate for reparametization
        beta: 1.0											# tradeoff of commitment loss and codebook loss
        use_learnable_std: True								# 
        use_ema_update:True									# use EMA update to learn codebook
        replace_freq: 100									# replace frequency
        norm: l2											# L2 normalization
        cb_norm: l2											# L2 normalization for codebook

data:
  dataset: cifar10
  batch_size: 64
  max_epoch: 90
  warm_epoch: 10

```



### VIT

TODO..



