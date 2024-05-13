from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from Network.Taming.models.vqgan import VQModel
from torchvision.datasets import ImageFolder

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping


class VQ_GAN_Trainer(object):
    def __init__(self, data_configs, model_configs):
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.model_params = self.model_configs['params']
        self.max_epoch = data_configs['max_epoch']
        self.warm_epoch = data_configs['warm_epoch']

    def get_network(self, archi='vqgan', pretrained_file=None):
        if archi == 'vqgan':
            model = VQModel(
                            self.model_params['ddconfig'],
                            self.model_params['lossconfig'],
                            self.model_configs['vq_params'],
                            self.model_configs['learning_rate'],
            )
        else:
            raise NotImplementedError
        return model

    def get_data_loader(self):
        transform = transforms.Compose([
                                        transforms.Resize(128),
                                        transforms.RandomCrop((128, 128)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        if self.data_configs['dataset'] == 'cifar10':
            train_dataset = CIFAR10(root='/home/zwh/data/cifar', train=True, download=True, transform=transform)
            test_dataset = CIFAR10(root='/home/zwh/data/cifar', train=False, download=True, transform=transform)
        elif self.data_configs['dataset'] == 'celeba':
            data = ImageFolder(root="/home/zwh/img_align_celeba", transform=transform)
            train_dataset, test_dataset =torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(0.8 * len(data))])           
        else:
            raise NotImplementedError
        train_loader = DataLoader(train_dataset, batch_size=self.data_configs['batch_size'], shuffle=True, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.data_configs['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        return train_loader, test_loader

    def fit(self):
        model = self.get_network()
        train_loader, test_loader = self.get_data_loader()

        early_stop_callback = EarlyStopping(monitor='val/perplexity', mode='max')

        model.set_iter_config(len(train_loader) * self.warm_epoch, len(train_loader) * self.max_epoch)
        logger = pl.loggers.TensorBoardLogger(save_dir=self.model_configs['log_dir'])
        trainer = pl.Trainer(
                            max_epochs=self.max_epoch,
                            accelerator='gpu',
                            devices=[0],
                            log_every_n_steps=100,
                            # check_val_every_n_epoch=2,
                            logger=logger,
                            callbacks=[early_stop_callback]
                            )
        trainer.fit(model, train_loader, test_loader)
        trainer.save_checkpoint(self.model_configs['save_path'] + '/last.ckpt')


    #
    # def eval(self):
    #
