import torch
import pytorch.ignite.metrics.FID as fid
import torchvision.transforms as transforms
import torchvision.models.inception_v3 as inception

from torchvision.datasets import ImageFolder
from omegaconf import OmegaConf
from Network.Taming.models.vqgan import VQModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


inception_model = inception(pretrained=True)
vqgan_folder = '/home/zwh/Modified_MaskGit/pretrained_maskgit/VQGAN_Celeba_Ori'

config = OmegaConf.load(vqgan_folder + "model.yaml")
model = VQModel(vqparams=config.model.vq_params, **config.model.params, )
checkpoint = torch.load(vqgan_folder + "last.ckpt", map_location="cpu")["state_dict"]
# Load network
model.load_state_dict(checkpoint, strict=False)
model = model.eval()
model = model.to("cuda")

#load dataset 
data=ImageFolder(
            root='/home/zwh/img_align_celeba', 
            transform=transforms.Compose([
                                            transforms.Resize(128),
                                            transforms.RandomCrop((128, 128)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                        ])
            )
data_train, data_test =torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(0.8 * len(data))])
test_loader = DataLoader(data_test, batch_size=128,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        )


for batch in test_loader:
    real_img = batch[0].to('cuda')
    reco_img, _, perplexity = model(real_img)
    fid.
    print('perplexity : {} | ')