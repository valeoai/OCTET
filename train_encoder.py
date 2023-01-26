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
from models import DecisionDensenetModel
from omegaconf import OmegaConf
from data.utils import CustomImageDataset

output_dir = 'encoder_training'
os.makedirs(output_dir, exist_ok=True)
device = 'cuda'
bs = 2
mlp_idx = -1
log_imgs_every = 500
save_every = 2_000
lr = 0.002
n_iters = 150000
blobgan_weights = 'checkpoints/blobgan_256x512.ckpt'
decision_model_weights = 'checkpoints/decision_densenet.tar'

config = OmegaConf.load('src/configs/experiment/invertblobgan_rect.yaml')
config.model.generator_pretrained = blobgan_weights
model = models.get_model(**config.model).to(device)

decision_model = DecisionDensenetModel(num_classes=4)
decision_model.load_state_dict(torch.load(decision_model_weights)['model_state_dict'])
decision_model.eval().to(device)
no_jiter = True
stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
aspect_ratio, resolution = config.model.generator.aspect_ratio, config.model.generator.resolution

if aspect_ratio != 1 and type(resolution) == int:
    resolution = (resolution, int(aspect_ratio*resolution))
transform = T.Compose([
    t for t in [
        T.Resize(resolution, T.InterpolationMode.LANCZOS),
        T.CenterCrop(resolution),
        #T.RandomHorizontalFlip(),
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

scheduler_steplr = StepLR(optimizer, step_size=5000, gamma=0.5)

losses_ = {'fake_MSE':[], 'fake_LPIPS':[], 'fake_decision':[], 'fake_latents_MSE':[],
           'real_MSE':[], 'real_LPIPS':[], 'real_decision':[], 'T_loss':[]}
iters = 0
best_loss = 100
while iters < n_iters:
    pbar = tqdm(dataloader_train, leave=True)#, desc=f'Image {i}')
    for batch_idx, batch in enumerate(pbar):
        batch_real, batch_labels = batch
        batch_real = batch_real.to(device)
        z = torch.randn(bs, model.generator.noise_dim).to(device)
        log_images = batch_idx % log_imgs_every ==0
        with torch.no_grad():
            layout_gt_fake, gen_imgs = model.generator.gen(z, ema=True, viz=log_images, no_jiter=no_jiter,
                                                            ret_layout=True, viz_colors=KLD_COLORS)
            target_decision_feat_fake = decision_model.feat_extract(gen_imgs)
            target_decision_feat_real = decision_model.feat_extract(batch_real)


        losses = dict()

        z_pred_fake = model.inverter(gen_imgs.detach())

        layout_pred_fake, reconstr_fake = model.generator.gen(z=z_pred_fake, ema=True, no_jiter=no_jiter, viz=log_images, ret_layout=True, mlp_idx=-1, viz_colors=KLD_COLORS)

        losses['fake_MSE'] = (gen_imgs - reconstr_fake).pow(2).mean()
        losses['fake_LPIPS'] = model.L_LPIPS(reconstr_fake, gen_imgs).mean()

        decision_feat_fake = decision_model.feat_extract(reconstr_fake)
        losses['fake_decision'] = torch.mean((decision_feat_fake - target_decision_feat_fake) ** 2)
        latent_l2_loss = []
        for k in ('xs', 'ys', 'covs', 'sizes', 'features', 'spatial_style'):
            latent_l2_loss.append((layout_pred_fake[k] - layout_gt_fake[k].detach()).pow(2).mean())
        losses['fake_latents_MSE'] = sum(latent_l2_loss) / len(latent_l2_loss)

        z_pred_real = model.inverter(batch_real.detach())
        layout_pred_real, reconstr_real = model.generator.gen(z_pred_real, ema=True, viz=log_images, ret_layout=True,
                                                                no_jiter=no_jiter, mlp_idx=-1, viz_colors=KLD_COLORS)
                                                                #mlp_idx=len(self.generator.layout_net_ema.mlp))

        losses['real_MSE'] = (batch_real - reconstr_real).pow(2).mean()
        losses['real_LPIPS'] = model.L_LPIPS(reconstr_real, batch_real).mean()

        decision_feat_real = decision_model.feat_extract(reconstr_real)
        losses['real_decision'] = torch.mean((decision_feat_real - target_decision_feat_real) ** 2)

        total_loss = f'T_loss'
        losses[total_loss] = sum(map(lambda k: losses[k] * model.λ[k], losses))

        log_message = ''
        for (key,val) in  losses.items():
            losses_[key].append(val.item())
            if len(losses_[key])>100:
                losses_[key].pop(0)
            short_key = key.replace('fake','F').replace('real','R').replace('decision', 'Dec').replace('latents','lat')
            log_message += f'{short_key}: {sum(losses_[key])/len(losses_[key]):.3f}, '

        pbar.set_description_str(log_message)
        # Do optimization.
        optimizer.zero_grad()
        losses[total_loss].backward()
        optimizer.step()
        scheduler_steplr.step()
        if log_images:
            with torch.no_grad():
                imgs = {
                    'real': batch_real,
                    'real_reconstr': reconstr_real,
                    'fake': gen_imgs,
                    'fake_reconstr': reconstr_fake,
                    'real_reconstr_feats': torch.nn.functional.interpolate(layout_pred_real['feature_img'], size=resolution),
                    'fake_reconstr_feats': torch.nn.functional.interpolate(layout_pred_fake['feature_img'], size=resolution),
                    'fake_feats': torch.nn.functional.interpolate(layout_gt_fake['feature_img'], size=resolution)
                }
                imgs = {k: v.clone().detach().float().cpu() for k, v in imgs.items()}
                imgs = torch.cat([v for v in imgs.values()], 0)
                image_grid = make_grid(
                    imgs, normalize=True, value_range=(-1, 1), nrow=bs
                )
                image_grid = F.to_pil_image(image_grid)

                image_grid = image_grid.save(f"{output_dir}/step_{iters}.jpg")
        iters += 1
        if iters % save_every == 0:
            state_dict = {'model': model.inverter.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler_steplr.state_dict(),
                        'iter': iters}
            torch.save(state_dict,f"{output_dir}/last.pt")
            mean_loss = sum(losses_[key])/len(losses_[key])
            if mean_loss < best_loss:
                best_loss = mean_loss
                torch.save(state_dict,f"{output_dir}/best.pt")
