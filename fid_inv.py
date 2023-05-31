import os
import sys
import argparse

import ipdb
import torch
import numpy as np
import torchvision.transforms as T

from tqdm import tqdm
from torchvision.transforms import functional as F

import lpips

from cleanfid.fid import frechet_distance
from cleanfid.features import (build_feature_extractor,
                               get_reference_statistics)

here_dir = '.'
sys.path.append(os.path.join(here_dir, 'src'))

from models import DecisionDensenetModel, load_model
from data.utils import CustomImageDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FID")

    parser.add_argument("--blobgan_weights", type=str, help="path to model checkpoint",
                        default='checkpoints/blobgan_256x512.ckpt'
    )

    parser.add_argument("--decision_model_weights", type=str, help="path to model decision model weights",
                        default='checkpoints/decision_densenet.tar'
    )
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument(
        "--dataset_name",
        type=str,
        default='bdd_rectangle_val_256',
        help="dataset name used when computing real imgs stats with setup_fid.py",
    )
    parser.add_argument(
        "--inv_path",
        type=str,
        default='shared_files/validation_rec_reproducible.pt',
        help="path to file containing blob parameters",
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
    blobgan_weights = args.blobgan_weights
    dataset_name = args.dataset_name
    inv_path = args.inv_path
    batch_size = args.batch_size
    dataset_path = args.dataset_path
    decision_model_weights = args.decision_model_weights


    logfile = open(inv_path.replace('.pt', '.txt'),"a")

    # \n is placed to indicate EOL (End of Line)

    torch.cuda.set_device(device)

    print(f'loading model from {blobgan_weights}')
    logfile.write(f'loading model from {blobgan_weights} \n')

    model = load_model(blobgan_weights, device)


        #decision model
    decision_model = DecisionDensenetModel(num_classes=4)
    decision_model.load_state_dict(torch.load(decision_model_weights)['model_state_dict'])
    decision_model.eval().to(device)

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

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    def lpips_loss(x1,x2):
        return loss_fn_vgg(x1, x2).squeeze()

    model.render_kwargs['norm_img'] = False

    model_feat = build_feature_extractor('clean', device)
    try:
        ref_mu, ref_sigma = get_reference_statistics(dataset_name, 256,
                                                    mode='clean', seed=0, split='custom')
    except:
        stats = np.load(f'fid_stats/{dataset_name}_clean_custom_na.npz')
        ref_mu, ref_sigma = stats["mu"], stats["sigma"]
    metadata = torch.load(inv_path)

    feats = []
    lpip = 0
    decision = np.zeros(4)
    i=0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            image_names= batch[1]
            real_imgs = batch[0].to(device)
            layout = [metadata[name] for name in image_names]
            layout = {k: torch.cat([layout[i][k] for i in range(batch_size)]) for k in layout[0].keys()}
            layout = {k:v.to(device) for k,v in layout.items()}

            rec_imgs = model.gen(layout=layout, ema=True, norm_img=False, no_jitter=True)

            distance = lpips_loss(real_imgs, rec_imgs).detach().cpu().numpy()
            lpip += np.nanmean(distance).item()

            initial_scores = decision_model(real_imgs)
            rec_scores = decision_model(rec_imgs)
            decision += (rec_scores.round()==initial_scores.round()).sum(0).cpu().numpy()

            rec_imgs = rec_imgs.add_(1).div_(2).mul_(255)
            rec_imgs = F.resize(rec_imgs, (299,299)).clip(0, 255)
            feat = model_feat(rec_imgs).detach().cpu().numpy()
            feats.append(feat)

            i+=1
    np_feats = np.concatenate(feats)

    print(f'LPIPS: {lpip/i}')
    logfile.write(f'LPIPS: {lpip/i} \n')
    print(f'Accuracy: {decision/(i*batch_size)}')
    logfile.write(f'Accuracy: {decision/(i*batch_size)} \n')

    v=np_feats[~np.isnan(np_feats).any(axis=1)] #TODO check why there are some nan values for some CFs
    mu = np.mean(v, axis=0)
    sigma = np.cov(v, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)

    print(f'fid score: {fid}')
    logfile.write(f'fid score: {fid} \n')
    logfile.close()

