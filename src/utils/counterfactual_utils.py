import numpy as np
import random
import torch

def get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return torch.round(tensor.cpu().detach().squeeze(), decimals=4).numpy()
 
def for_canvas_(img):
    if img.ndim==4:
        return img.round().permute(0, 2, 3, 1).clamp(min=0, max=255).cpu().numpy().astype(np.uint8)
    return img.round().permute(1, 2, 0).clamp(min=0, max=255).cpu().numpy().astype(np.uint8)

def generate_layout_feat(model, z, truncate=None, mlp_idx=None):
  num_features = random.randint(model.n_features_min, model.n_features_max)
  if truncate is not None:
    mlp_idx = -1
    z = model.layout_net_ema.mlp[:mlp_idx](z)
    z = (model.mean_latent * truncate) + (z * (1 - truncate))
  return model.layout_net_ema(z, num_features, mlp_idx)

def opt_var(params, target_params=None):
  '''
  sets requires_grad True for target params
  '''
  opt_params = {}
  req_grad = False
  for key, val in params.items():
    if target_params is not None:
      req_grad = (key in target_params)
    opt_params[key] = val.clone().detach().requires_grad_(req_grad)
  return opt_params

#### for blob targeting optim ###
def devide_layout(layout, target_blobs):
    target_blob_background = [t+1 for t in target_blobs]
    target_layout = {}
    for key,val in layout.items():
        if key in ['xs', 'ys', 'covs']:
            target_layout[key] = val[:, target_blobs]
        else:
            target_layout[key] = val[:, target_blob_background]
    return target_layout

#### for blob targeting optim ###
def assemble(target_layout, layout, target_blobs):
    target_blob_background = [t+1 for t in target_blobs]
    layout_ = {}
    for key in layout.keys():
        layout_[key] = layout[key].detach().clone()
        if key in ['xs', 'ys', 'covs']:
            layout_[key][:, target_blobs] = target_layout[key]
        else:
            layout_[key][:, target_blob_background] = target_layout[key]
    return layout_