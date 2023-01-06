import sys
from typing import Any, Tuple, Dict
import torch
from pytorch_lightning import LightningModule

from models import networks
from utils import to_dataclass_cfg
# from .segmenter import *
from .blobgan import *
from .gan import *
from .invertblobgan import *
from .decision_models import DecisionResnetModel, DecisionDensenetModel, DecisionExplainDensenetModel
import sys
sys.path.append("..")
from utils import KLD_COLORS, viz_score_fn

def get_model(name: str, return_cfg: bool = False, **kwargs) -> Tuple[LightningModule, Dict[str, Any]]:
    cls = getattr(sys.modules[__name__], name)
    cfg = to_dataclass_cfg(kwargs, cls)
    if return_cfg:
        return cls(**cfg), cfg
    else:
        return cls(**cfg) 

def load_model(path, device='cuda'):
    model = BlobGAN.load_from_checkpoint(path, strict=False).to(device)
    COLORS = KLD_COLORS
    model.colors = COLORS
    size= (model.generator.size_in,int(model.aspect_ratio*model.generator.size_in))
    torch.manual_seed(0)
    noise = [torch.randn((1, 1, size[0] * 2 ** ((i + 1) // 2), size[1] * 2 ** ((i + 1) // 2))).to(device) for i in
             range(model.generator_ema.num_layers)]

    model.noise = noise
    render_kwargs = {
        'no_jitter': True,
        'ret_layout': True,
        'viz': True,
        'ema': True,
        'viz_colors': COLORS,
        'norm_img': True,
        'viz_score_fn': viz_score_fn,
        'noise': noise
    }
    model.render_kwargs = render_kwargs
    print('\033[92m' 'Done loading and configuring model!', flush=True)
    return model