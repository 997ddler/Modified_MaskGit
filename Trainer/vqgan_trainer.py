from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from Network.Taming.models.vqgan import VQModel

import pytorch_lightning as pl


class VQ_GAN_Trainer(object):
    def __init__(self, data_configs, model_configs):
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.model_params = self.model_configs['params']
        self.max_epoch = data_configs['max_epoch']

    def get_network(self, archi='vqgan', pretrained_file=None):
        if archi == 'vqgan':
            model = VQModel(
                            self.model_params['ddconfig'],
                            self.model_params['lossconfig'],
                            self.model_params['n_embed'],
                            self.model_params['embed_dim'],
                            self.model_configs['learning_rate'],
            )
        else:
            raise NotImplementedError
        return model

    def get_data_loader(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((32, 32), antialias=None)])
        if self.data_configs['dataset'] == 'cifar10':
            train_dataset = CIFAR10(root='~/data/cifar', train=True, download=True, transform=transform)
            test_dataset = CIFAR10(root='~/data/cifar', train=False, download=True, transform=transform)
        else:
            raise NotImplementedError
        train_loader = DataLoader(train_dataset, batch_size=self.data_configs['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.data_configs['batch_size'], shuffle=True)
        return train_loader, test_loader

    def fit(self):
        model = self.get_network()
        train_loader, test_loader = self.get_data_loader()
        trainer = pl.Trainer(max_epochs=self.max_epoch)
        trainer.fit(model, train_loader, test_loader)
        trainer.save_checkpoint('D:/discrete representation/Maskgit-pytorch/pretrained_maskgit/VQGAN/last.ckpt')


    #
    # def eval(self):
    #
