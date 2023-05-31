import os
import sys
import random
import numpy as np
from collections import defaultdict

import pickle

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

# Load encoder (model.inverter) and generator
encoder_path = "encoder_pretraining"
encoder_path = "encoder_finetuning"
encoder_path = "encoder_finetuning_l1_lpips"
encoder_path = "encoder_finetuning_no_decision"

config = OmegaConf.load('src/configs/experiment/invertblobgan_rect.yaml')
device = 'cuda'
blobgan_weights = 'checkpoints/blobgan_256x512.ckpt'
decision_model_weights = 'checkpoints/decision_densenet.tar'

config.model.generator_pretrained = blobgan_weights
model = models.get_model(**config.model).to(device)
model.inverter.load_state_dict(torch.load('%s/best.pt' % encoder_path)['model']) # Load pretrained encoder

# Load decision model
decision_model = DecisionDensenetModel(num_classes=4)
decision_model.load_state_dict(torch.load(decision_model_weights)['model_state_dict'])
decision_model.eval().to(device)

# Load val dataset
dataset_path = '/datasets_local/BDD/bdd100k/images/100k/val'
bs = 2

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

dataset = CustomImageDataset(dataset_path, transform)
dataloader_val = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False, drop_last=False)

# Forward images
pbar = tqdm(dataloader_val, leave=True)

results = {}

# Iterate over batches
for batch_idx, batch in enumerate(pbar):
    batch_real, batch_labels = batch
    batch_real = batch_real.to(device)

    with torch.no_grad():

        # Encode images
        z_pred_real = model.inverter(batch_real.detach())
        # Decode encoded images
        layout_pred_real, reconstr_real = model.generator.gen(z_pred_real, ema=True, viz=False, ret_layout=True,
                                                                no_jiter=no_jiter, mlp_idx=-1, viz_colors=KLD_COLORS)

        # LPIPS between real and reconstructed
        lpips = model.L_LPIPS(reconstr_real, batch_real)

        # L2 between real and reconstructed
        l2 = (batch_real - reconstr_real).pow(2)

        # Decision real
        decisions_real = decision_model(batch_real)

        # Decision reconstructed
        decisions_reconstructed = decision_model(reconstr_real)

    # Store values in results
    for i, im_name in enumerate(batch_labels):
        results[im_name] = {}
        results[im_name]["lpips"] = lpips[i, 0, 0, 0].detach().cpu().numpy().tolist()
        results[im_name]["l2"] = l2[i].mean().detach().cpu().numpy().tolist()
        results[im_name]["decisions_real"] = decisions_real[i].detach().cpu().numpy()
        results[im_name]["decisions_reconstruction"] = decisions_reconstructed[i].detach().cpu().numpy()

# Save results
with open("/root/workspace/OCTET/%s/val_images_results.pkl" % encoder_path, "wb") as f:
    pickle.dump(results, f)

# # Load results
# with open("/root/workspace/OCTET/%s/val_images_results.pkl" % encoder_path, "rb") as f:
#     results = pickle.load(f)

# Analyze results
lpips = []
l2 = []
accuracy = defaultdict(list)
avg_acc = []

for im_name, im_scores in results.items():
    lpips.append(im_scores["lpips"])
    l2.append(im_scores["l2"])

    decision_real = im_scores["decisions_real"]
    decision_reconstruction = im_scores["decisions_reconstruction"]
    # binarize
    d_r = np.rint(decision_real)
    d_rec = np.rint(decision_reconstruction)
    # Check if same decision
    for i in range(4):
        accuracy[i].append(d_r[i] == d_rec[i])
        avg_acc.append(d_r[i] == d_rec[i])

print("lpips \t\t", "%.3f" % (np.mean(lpips)))
print("l2 \t\t", "%.3f" % (np.mean(l2)))
for i in range(4):
    print("accuracy %s \t" % i, np.mean(accuracy[i]))
print("accuracy AVG \t", np.mean(avg_acc))

