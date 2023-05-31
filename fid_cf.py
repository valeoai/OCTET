import argparse
import os, sys
import torch
from tqdm import tqdm
import ipdb
import numpy as np
from torchvision.transforms import functional as F
from cleanfid.features import build_feature_extractor, get_reference_statistics
from cleanfid.fid import frechet_distance

here_dir = '.'
sys.path.append(os.path.join(here_dir, 'src'))

from models import load_model



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

    args = parser.parse_args()

    device = args.device
    blobgan_weights = args.blobgan_weights
    dataset_name = args.dataset_name
    metadata_folder = args.metadata_folder

    metadat_files = os.listdir(metadata_folder)
    metadat_files = [f for f in metadat_files if f.endswith('.pt')]

    logfile = open(f"{metadata_folder}/metrics.txt", "a")

    # \n is placed to indicate EOL (End of Line)
    logfile.write("FID \n")

    torch.cuda.set_device(device)
    print(f'experiment {metadata_folder}')
    logfile.write(f'experiment {metadata_folder} \n')

    print(f'loading model from {blobgan_weights}')
    logfile.write(f'loading model from {blobgan_weights} \n')

    model = load_model(blobgan_weights, device)

    feats = {'CF0':[], 'CF1':[], 'CF2':[], 'CF3':[]}
    model_feat = build_feature_extractor('clean', device)
    try:
        ref_mu, ref_sigma = get_reference_statistics(dataset_name, 256,
                                                    mode='clean', seed=0, split='custom')
    except:
        stats = np.load(f'fid_stats/{dataset_name}_clean_custom_na.npz')
        ref_mu, ref_sigma = stats["mu"], stats["sigma"]
    with torch.no_grad():
        for f in tqdm(metadat_files):
            f = os.path.join(metadata_folder, f)
            metadata = torch.load(f)
            for i in range(4):
                layout = metadata[f'att_{i}']['blob_cf']
                layout = {k:v.to(device) for k,v in layout.items()}
                CF_images = model.gen(layout=layout, ema=True, norm_img=True, no_jitter=True)
                CF_images = F.resize(CF_images, (299,299)).clip(0, 255)
                feat = model_feat(CF_images).detach().cpu().numpy()
                feats[f'CF{i}'].append(feat)
        np_feats = {k:np.concatenate(v) for k,v in feats.items()}

        for k,v in np_feats.items():
            v=v[~np.isnan(v).any(axis=1)] #TODO check why there are some nan values for some CFs
            mu = np.mean(v, axis=0)
            sigma = np.cov(v, rowvar=False)
            print("computing fid...")
            fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)

            print(f'fid score {k}: {fid}')
            logfile.write(f'fid score {k}: {fid} \n')

    logfile.close()

