# Borrowed from https://github.com/nicolas-dufour/diffusion/blob/master/metrics/sample_and_eval.py
import random
import clip
import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.utils import save_image
from Metrics.inception_metrics import MultiInceptionMetrics
from torchvision.utils import make_grid
from abc import ABC, abstractmethod

def remap_image_torch(image):
    min_norm = image.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
    max_norm = image.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
    image_torch = ((image - min_norm) / (max_norm - min_norm)) * 255
    image_torch = torch.clip(image_torch, 0, 255).to(torch.uint8)
    return image_torch

class SampleAndEval(ABC):
    """
        The parent class which is template to compute some metrcis
        For VQ-VAE, compute rFID score and save reconstructions
        For VIT, compute FID and then compute FID* (FID* is metric calculated by pytorch-fid)
        params:
            device -> int: select machine to compute
            num_images -> int: the maximum images used when computer metrics
            compute_per_class_metrics -> bool: whether compute different class's FID score
            num_classes -> int: number of classes
            use_label -> bool: select conditional FID or unconditional FID
        return:
    """
    def __init__(self, device, num_images=5000, compute_per_class_metrics=False, num_classes=10, use_label=True):
        super().__init__()
        self.inception_metrics = MultiInceptionMetrics(
            reset_real_features=False,
            compute_unconditional_metrics= not use_label,
            compute_conditional_metrics=use_label,
            compute_conditional_metrics_per_class=compute_per_class_metrics,
            num_classes=num_classes,
            num_inception_chunks=10,
            manifold_k=3,
        ).to(device)
        self.num_images = num_images
        self.true_features_computed = False
        self.device = device
        
    @abstractmethod
    def compute_and_log_metrics(self, module):
        pass
    
    @abstractmethod
    def compute_true_images_features(self, dataloader):
        pass
    
    @abstractmethod
    def compute_fake_images_features(self, module, dataloader):
        pass

class SampleAndEvalVIT(SampleAndEval):
    """
        The specific class to compute FID and FID *
        For VIT, compute FID and then compute FID* (FID* is metric calculated by pytorch-fid)
        And we also save images used for FID and FID*
        params:
            device -> int: select machine to compute
            num_images -> int: the maximum images used when computer metrics
            compute_per_class_metrics -> bool: whether compute different class's FID score
            num_classes -> int: number of classes
            use_label -> bool: select conditional FID or unconditional FID
        return:
    """
    def __init__(self, device, num_images=5000, compute_per_class_metrics=False, num_classes=10, use_label=True):
        super().__init__(device, num_images=num_images, compute_per_class_metrics=False, num_classes=num_classes, use_label=use_label)
        self.save_images_dir = ''

    def compute_and_log_metrics(self, module):
        """
        The top module to invoke different method
        params:
            module -> MaskGIT: the module contains VIT and dataloader
        return:
        """
        with torch.no_grad():
            if not self.true_features_computed or not self.inception_metrics.reset_real_features:
                self.compute_true_images_features(module.test_data)
                self.true_features_computed = True
            self.compute_fake_images_features(module, module.test_data)

            metrics = self.inception_metrics.compute()
            metrics = {f"Eval/{k}": v for k, v in metrics.items()}
            print(metrics)



    def compute_true_images_features(self, dataloader):
        """ computing features of true images
            params:
                dataloader -> Dataloder: the dataloader used for VIT
            return:
        """
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images
        bar = tqdm(dataloader, leave=False, desc="Computing true images features")
        for i, (images, labels) in enumerate(bar):
            if i * dataloader.batch_size >= max_images:
                break

            self.inception_metrics.update(remap_image_torch(images.to(self.device)),
                                          labels.to(self.device),
                                          image_type="real")

    def compute_fake_images_features(self, module, dataloader):
        """ computing features of generated images
            params:
                module     -> MaskgitTransformer: VIT
                dataloader -> Dataloder: the dataloader used for VIT
            return:
        """
        if len(dataloader.dataset) < self.num_images:
            max_images = len(dataloader.dataset)
        else:
            max_images = self.num_images

        bar = tqdm(dataloader, leave=False, desc="Computing fake images features")
        for i, (images, labels) in enumerate(bar):
            if i * dataloader.batch_size >= max_images:
                break

            with torch.no_grad():
                # if use_label == True, the input of model should contains labels
                if self.use_label:
                    if isinstance(labels, list):
                        labels = clip.tokenize(labels[random.randint(0, 4)]).to(self.device)
                        labels = module.clip.encode_text(labels).float()
                    else:
                        labels = labels.to(self.device)
                else:
                    labels = None
                    
                # sample
                images = module.sample(nb_sample=images.size(0),
                                       labels=labels,
                                       sm_temp=module.args.sm_temp,
                                       w=module.args.cfg_w,
                                       randomize="linear",
                                       r_temp=module.args.r_temp,
                                       sched_mode=module.args.sched_mode,
                                       step=12
                                       )[0]
                
                # save images to compute FID*
                if self.save_images_dir != '':
                    output_folder = self.save_images_dir
                    os.makedirs(output_folder, exist_ok=True)

                    # save image
                    for j in range(images.size(0)):
                        image_tensor = images[j]
                        save_image(image_tensor, os.path.join(output_folder, f'image_{i * dataloader.batch_size + j+1}.png'), normalize=True)

                # compute FID score
                images = images.float()
                # print(f"Saved {tensor.size(0)} images to '{output_folder}'")
                image_type = "conditional" if self.use_label else "unconditional"
                self.inception_metrics.update(remap_image_torch(images),
                                              labels=labels,
                                              image_type=image_type
                                              )
                
    def set_save_directory(self, target : str):
        """set path to save generated pictures to compute FID* """
        self.save_images_dir = target
        
    def create_combined_image(self, num_images=32, img_size=(128, 128)):
        """ to clearly visulize generated images as grid form
            the new grid images will be saved on the last directory of whole generated images
            params:
                    num_images -> int: the number of images to be shown in a grid
                    img_size   -> tuple: the size of image
            return:
        """
        folder_path = self.save_images_dir
        image_files = []
        for i, path in enumerate(os.listdir(folder_path)):
            image_files.append(path)
            if i == 31:
                break
        
        images = [Image.open(os.path.join(folder_path, img)).resize(img_size) for img in image_files]
        
        grid_cols = 8
        grid_rows = 4
        
        fig, axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols*img_size[0]//100, grid_rows*img_size[1]//100))
        fig.subplots_adjust(hspace=0, wspace=0)
        
        for ax, img in zip(axs.flatten(), images):
            ax.imshow(img)
            ax.axis('off')
        
        last_dir_folder= os.path.join(folder_path, "..")
        output_file = os.path.join(last_dir_folder, f"{os.path.basename(folder_path)}.jpg")
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Combined image saved as {output_file}")
        
    def compute_FID_star(self):
        generated_images_folder = self.save_images_dir
        inception_model = torchvision.models.inception_v3(weights=Inception_V3_Weights.DEFAULT)

        # 计算FID距离值
        fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
                                                    #inception_model,
                                                    128,
                                                    device="cuda",
                                                    dims=2048
        )

class SampleAndEvalVQ(SampleAndEval):    
    """
        The specific class to compute rFID and save reconstruction images
        params:
            device -> int: select machine to compute
            num_images -> int: the maximum images used when computer metrics
            compute_per_class_metrics -> bool: whether compute different class's FID score
            num_classes -> int: number of classes
            use_label -> bool: select conditional FID or unconditional FID
        return:
    """
    def __init__(self,device, num_images=5000, compute_per_class_metrics=False, num_classes=10, use_label=True) -> None:
        super().__init__(device, num_images=5000, compute_per_class_metrics=False, num_classes=10, use_label=use_label)
        
    def compute_and_log_metrics(self, module):
        """compute and log FID score and save reconstructions pictures
            :param
                module -> VQ_VAE_Trainer: the model of VQVAE and its dataloader
            :return:
        """
        # compute FID score
        with torch.no_grad():
            self.compute_true_images_features(module.test_loader)
            self.compute_fake_images_features(module.model, module.test_loader)
            results = self.inception_metrics.compute()
            metrics = {f"Eval/{k}": v for k, v in results.items()}
            print(metrics)
            
            # save reconstructions
            model_name = module.model_configs['model_name']
            save_path = module.model_configs['save_path']
            self.get_reconstruction(module.model, module.test_loader, model_name, save_path)

    def compute_true_images_features(self, dataloader):
        """ computing features of true images
            params:
                dataloader -> Dataloder: the dataloader used for VQ-VAE
            return:
        """
        if len(dataloader.dataset) < self.num_images:
            num = len(dataloader.dataset)
        else:
            num = self.num_images
        bar = tqdm(dataloader, leave=False, desc='computing true images features')
        for i, (images, _) in enumerate(bar):
            if i * dataloader.batch_size >= num:
                break

            images = images.to('cuda')
            self.inception_metrics.update(remap_image_torch(images),image_type="real")

    def compute_fake_images_features(self, module, dataloader):
        """ computing features of reconstructed images
            params:
                module     -> VQModel: VQ-VAE
                dataloader -> Dataloder: the dataloader used for VQ-VAE
            return:
        """
        if len(dataloader.dataset) < self.num_images:
            num = len(dataloader.dataset)
        else:
            num = self.num_images

        bar = tqdm(dataloader, leave=False, desc="Computing fake images features")
        for i, (images, _) in enumerate(bar):
            if i * dataloader.batch_size >= num:
                break
            
            images = images.to('cuda')
            with torch.no_grad():
                images, _, _ = module(images)
                images = images.float()
                self.inception_metrics.update(remap_image_torch(images), image_type="unconditional")

    def get_reconstruction(self, model, dataloader, model_name, save_path):
        """ get reconstructed images and save them
            params: 
                model      -> VQModel: VQ-VAE
                dataloader -> Dataloder: the dataloader used for VIT
                model_name -> str: the name of model
                save_path  -> directory to be saved
            return:
        """
        
        batch = next(iter(dataloader))[0][32 : ]
        reco, _, _  = model(batch.cuda())
        ori = make_grid(batch, 8, padding=2, normalize=True).permute(1, 2, 0).cpu().numpy()
        reco = make_grid(reco.float(), 8, padding=2, normalize=True).permute(1, 2, 0).cpu().numpy()
        
        plt.axis('off')
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(ori)
        ax1.set_title('Original')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(reco)   
        ax2.set_title('Reconstruction')
        path = os.path.join(
            save_path,
            model_name + '.png'
        )
        plt.savefig(path, dpi=500)
        print('images already be saved to ' + path)
