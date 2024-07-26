import torch
import pytorch_lightning as pl
from Metrics.sample_and_eval import SampleAndEvalVQ
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from Network.Taming.models.vqgan import VQModel
from torchvision.datasets import ImageFolder
from pytorch_lightning.callbacks import EarlyStopping

class VQ_VAE_Trainer(object):
    """ Initialization the trainer of VQ-VAE
        :params:
            data_configs  ->    dict: the configuration of data including dataset type, batch_size, max_epoch, warm_epoch
            model_configs ->    dict: the configuration of model including configuration of backbone and VQ
            test_vqvae    ->    bool: set this param true, we will only load model from save path and test model to get rFID(reconstruction FID score)
            device_id     ->    int: select the device to trian the model
    """
    def __init__(self, data_configs, model_configs, device_id=0, test_vqvae=False):
        self.data_configs = data_configs
        self.model_configs = model_configs
        
        # the params for VQ part
        self.model_params = self.model_configs['params']
        
        # set max training epoch
        self.max_epoch = data_configs['max_epoch']
        
        # set warm epoch 
        self.warm_epoch = data_configs['warm_epoch']
        
        # if test_vqvae == True only load model and test it
        self.test_vqvae = test_vqvae
        self.device_id = device_id
        
        # get the Sample and Evaluate Class to process Evaluation
        # For maskgit, this model provide perplexity, MSE, Distances and active ratio, FID score and Reconstruction of pictures.
        self.sae = SampleAndEvalVQ(device=device_id, num_images=20000, use_label=(data_configs['dataset'] == 'cifar10'))


    def get_network(self, archi='vqvae'):
        """ return the network, load checkpoint if self.test_vqvae == True
            :param
                archi -> str: vqvae the architecture to load
            :return
                model -> nn.Module: the network
        """
        if archi == 'vqvae':
            model = VQModel(
                            self.model_params['ddconfig'],
                            self.model_params['lossconfig'],
                            self.model_configs['vq_params'],
                            self.model_configs['learning_rate'],
            )
            if self.test_vqvae:
                # Load network from save path
                checkpoint = torch.load(self.model_configs['save_path'] + '/last.ckpt', map_location="cpu")["state_dict"]
                model.load_state_dict(checkpoint, strict=True)
                model = model.eval()
                model = model.to('cuda')    
        else:
            raise NotImplementedError
        return model

    def get_data_loader(self):
        """ return dataloader according to setting in data_configs. (cifar10 or celeba)
            :param
            :return
                train_loader -> torch.utils.data.DataLoader training dataset
                test_loader  -> torch.utils.data.DataLoader test dataset
                val_loader   -> torch.utils.data.DataLoader validation dataset for cifar10, it do not have validation dataset.
        """
        
        if self.data_configs['dataset'] == 'cifar10':
            transform=transforms.Compose([
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = CIFAR10(root='/home/zwh/data/cifar', train=True, download=True, transform=transform)
            test_dataset = CIFAR10(root='/home/zwh/data/cifar', train=False, download=True, transform=transform)
            val_dataset = None

        elif self.data_configs['dataset'] == 'celeba':
            transform = transforms.Compose([
                                transforms.Resize(128),
                                transforms.RandomCrop((128, 128)),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])  
            train_dataset = ImageFolder(root="/data/zwh/celeba_train_folder", transform=transform)
            val_dataset = ImageFolder(root="/data/zwh/celeba_val_folder", transform=transform)
            test_dataset = ImageFolder(root="/data/zwh/celeba_test_folder", transform=transform)      
        else:
            raise NotImplementedError
        
        train_loader = DataLoader(train_dataset, batch_size=self.data_configs['batch_size'], shuffle=True, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=self.data_configs['batch_size'], shuffle=False, num_workers=8)
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.data_configs['batch_size'], shuffle=False, num_workers=8)
        else:
            val_loader = None
        return train_loader, test_loader, val_loader
        
    def fit(self):
        """ The whole process to use different methods to train and test model."""
        # get network
        self.model = self.get_network()
        
        # get dataloader
        self.train_loader, self.test_loader, self.val_loader = self.get_data_loader()
        
        # set early stop if early_top == True the training will use early stop
        call_backs = []  if self.val_loader is None or not self.data_configs['early_stop'] else [EarlyStopping(monitor='val/recloss', mode='min')]
        
        # compute the warm iterations of training and set it.
        # For default setting, we use cosine scheduler 
        self.model.set_iter_config(len(self.train_loader) * self.warm_epoch, len(self.train_loader) * self.max_epoch)
        
        # get logger and trainer of model
        logger = pl.loggers.TensorBoardLogger(save_dir=self.model_configs['log_dir'], version=self.model_configs['model_name'])
        trainer = pl.Trainer(
                            max_epochs=self.max_epoch,
                            accelerator='gpu',
                            devices=[self.device_id],
                            log_every_n_steps=1000,
                            check_val_every_n_epoch=2,
                            logger=logger,
                            callbacks=call_backs
                            )
        
        # begin training
        if not self.test_vqvae:
            trainer.fit(self.model, self.train_loader, self.val_loader)
            trainer.save_checkpoint(self.model_configs['save_path'] + '/last.ckpt')
        
        # test
        trainer.test()
        self.model.eval()
        self.sae.compute_and_log_metrics(self)