import os
import sys
import argparse

import torch
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm

import lpips

from cleanfid.features import (build_feature_extractor,
                               get_reference_statistics)

here_dir = '.'
sys.path.append(os.path.join(here_dir, 'src'))

from models import load_model
from data.utils import CustomImageDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID")

    parser.add_argument("--blobgan_weights", type=str, help="path to model checkpoint",
                        default='checkpoints/blobgan_256x512.ckpt'
    )
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='bdd_rectangle_val_256',
        help="dataset name used when computing real imgs stats with setup_fid.py",
    )
    parser.add_argument(
        "--metadata_folder",
        type=str,
        default='experiments/style_structure/lambda0',
        help="experiment folder path containing metadata files",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='/datasets_local/BDD/bdd100k/seg/images/val'
    )
    parser.add_argument(
        '-bs', '--batch_size', default=4,
        help='Number of images to analyze in one forward pass. Adjust based on metadata folder.',
        type=int)
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    blobgan_weights = args.blobgan_weights
    metadata_folder = args.metadata_folder

    device = args.device
    torch.cuda.set_device(device)
    dataset_path = args.dataset_path

    metadat_files = os.listdir(metadata_folder)
    metadat_files = [f for f in metadat_files if f.endswith('.pt')]
    metadat_files = sorted(metadat_files, key=lambda f: int(f.split('.pt')[0].split('metadata_')[-1]))
    logfile = open(f"{metadata_folder}/metrics.txt","a")
    logfile.write("LPIPS and success rate \n")
    logfile.write(f'experiment {metadata_folder} \n')
    logfile.write(f'loading model from {blobgan_weights} \n')

    model = load_model(blobgan_weights, device)

    stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
    aspect_ratio, resolution = model.generator_ema.aspect_ratio, model.resolution

    if aspect_ratio != 1 and type(resolution) == int:
        resolution = (resolution, int(aspect_ratio*resolution))
    transform = T.Compose([
        t for t in [
            T.Resize(resolution, T.InterpolationMode.LANCZOS),
            T.CenterCrop(resolution),
            T.ToTensor(),
            T.Normalize(stats['mean'], stats['std'], inplace=True),
        ]
    ])
    dataset = CustomImageDataset(dataset_path, transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.render_kwargs['norm_img'] = False
    l_feats = []
    model_feat = build_feature_extractor('clean', device)
    iterator=iter(dataloader)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    def lpips_loss(x1,x2):
        return loss_fn_vgg(x1, x2).squeeze()

    lpip = [0,0,0,0]
    flip_ = [0,0,0,0]
    pbar = tqdm(metadat_files)
    i = 0
    with torch.no_grad():
        for f in pbar:
            batch = next(iterator)
            realbatch = batch[0].cuda()
            f = os.path.join(metadata_folder, f)
            metadata = torch.load(f)
            image_names = metadata['image_names']
            assert (list(batch[1]) == list(image_names)), (f'source and target images are not the same'
                                                f'Found{batch[1]} and {image_names}')

            init_scores = metadata['init_scores']
            for k in range(4):
                layout = metadata[f'att_{k}']['blob_cf']
                layout = {k:v.to(device) for k,v in layout.items()}
                _, img_batch = model.gen(layout=layout, **model.render_kwargs)
                distance = lpips_loss(img_batch, realbatch).detach().cpu().numpy()
                lpip[k] += np.nanmean(distance)

                scores_cf = metadata[f'att_{k}']['final_scores']
                flip = ((init_scores[:,k]) > .5 * (scores_cf[:,k] < .5)) + ((init_scores[:,k] < .5) * (scores_cf[:,k] > .5))
                flip_[k] += flip.sum()
            i += 1
    print(f'Success: {[l/(batch_size*i) for l in flip_]}')
    print(f'LPIPS: {[l/i for l in lpip]}')
    logfile.write(f'Success: {[l/(batch_size*i) for l in flip_]} \n')
    logfile.write(f'LPIPS: {[l/i for l in lpip]} \n')
