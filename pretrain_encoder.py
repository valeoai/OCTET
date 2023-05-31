import os
import sys
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from tqdm import tqdm
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import StepLR

here_dir = '.'
sys.path.append(os.path.join(here_dir, 'src'))

import models

from utils import KLD_COLORS
from omegaconf import OmegaConf
from data.utils import CustomImageDataset

output_dir = 'encoder_pretraining'
os.makedirs(output_dir, exist_ok=True)
device = 'cuda'
#bs = 8
bs = 16
mlp_idx = -1 #
log_imgs_every = 500
save_every = 1000
lr = 0.005
n_iters = 150000
blobgan_weights = 'checkpoints/blobgan_256x512.ckpt'

config = OmegaConf.load('src/configs/experiment/invertblobgan_rect.yaml')
config.model.generator_pretrained = blobgan_weights
model = models.get_model(**config.model).to(device)
no_jiter = True
stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
aspect_ratio, resolution = config.model.generator.aspect_ratio, config.model.generator.resolution

if aspect_ratio != 1 and type(resolution) == int:
    resolution = (resolution, int(aspect_ratio*resolution))
transform = T.Compose([
    t for t in [
        T.Resize(resolution, T.InterpolationMode.LANCZOS),
        T.CenterCrop(resolution),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(stats['mean'], stats['std'], inplace=True),
    ]
])
dataset_path = '/datasets_local/BDD/bdd100k/images/100k/train'
dataset = CustomImageDataset(dataset_path, transform)
dataloader_train = torch.utils.data.DataLoader(dataset, batch_size = bs, shuffle = True, drop_last=True)


params = list(model.inverter.parameters())
print(f'Optimizing {sum([p.numel() for p in params]) / 1e6:.2f}M params')
optimizer = torch.optim.Adam(params, lr=lr)

scheduler_steplr = StepLR(optimizer, step_size=2500, gamma=0.5)

losses_ = []
best_loss = 100
criterion = torch.nn.MSELoss()

pbar = tqdm(range(n_iters), leave=True)#, desc=f'Image {i}')
for batch_idx in pbar:
    z = torch.randn(bs, model.generator.noise_dim).to(device)
    log_images = batch_idx % log_imgs_every == 0
    with torch.no_grad():
        layout_gt_fake, gen_imgs = model.generator.gen(z, ema=True, viz=log_images, no_jiter=no_jiter,
                                                        ret_layout=True, viz_colors=KLD_COLORS)
        z = model.generator.layout_net_ema.mlp[:mlp_idx](z)


    z_pred_fake = model.inverter(gen_imgs.detach())

    with torch.no_grad():
        layout_pred_fake, reconstr_fake = model.generator.gen(z=z_pred_fake, ema=True, no_jiter=no_jiter, viz=log_images, ret_layout=True, mlp_idx=-1, viz_colors=KLD_COLORS)
    loss = criterion(z, z_pred_fake)

    log_message = ''
    losses_.append(loss.item())
    if len(losses_) > 100:
        losses_.pop(0)
    log_message += f'loss: {sum(losses_)/len(losses_):.4f}'

    pbar.set_description_str(log_message)
    # Do optimization.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler_steplr.step()
    if log_images:
        with torch.no_grad():
            imgs = {
                'fake': gen_imgs,
                'fake_reconstr': reconstr_fake,
                'fake_feats': torch.nn.functional.interpolate(layout_gt_fake['feature_img'], size=resolution),
                'fake_reconstr_feats': torch.nn.functional.interpolate(layout_pred_fake['feature_img'], size=resolution),
            }
            imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items()}
            imgs = torch.cat([v for v in imgs.values()], 0)
            image_grid = make_grid(
                imgs, normalize=True, value_range=(-1, 1), nrow=bs
            )
            image_grid = F.to_pil_image(image_grid)

            image_grid = image_grid.save(f"{output_dir}/step_{batch_idx}.jpg")
        print("loss", sum(losses_)/len(losses_))
    if (batch_idx+1) % save_every == 0 :
        state_dict = {'model': model.inverter.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler_steplr.state_dict(),
                    'iter': batch_idx}
        torch.save(state_dict,f"{output_dir}/last.pt")
        mean_loss = sum(losses_)/len(losses_)
        if mean_loss < best_loss:
            best_loss = mean_loss
            print("saving best at step", batch_idx)
            torch.save(state_dict,f"{output_dir}/best.pt")
