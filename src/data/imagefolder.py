from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union, Dict

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from data.nodata import NullIterableDataset
from data.utils import CustomImageDataset
from utils import print_once

_all__ = ['ImageFolderDataModule']


@dataclass
class ImageFolderDataModule(LightningDataModule):
    path: Union[str, Path]  # Root
    dataloader: Dict[str, Any]
    resolution: Union[int,tuple] = 256  # Image dimension
    aspect_ratio: float = 1.
    def __post_init__(self):
        super().__init__()
        self.path = Path(self.path)
        self.stats = {'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5)}
        if self.aspect_ratio != 1 and type(self.resolution) == int:
            self.resolution = (self.resolution, int(self.aspect_ratio*self.resolution))
        self.transform = T.Compose([
            t for t in [
                T.Resize(self.resolution, InterpolationMode.LANCZOS),
                T.CenterCrop(self.resolution),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.stats['mean'], self.stats['std'], inplace=True),
            ]
        ])
        self.data = {}

    def setup(self, stage: Optional[str] = None):
        for split in ('train', 'val', 'test'):
            path = self.path / split
            empty = True
            if path.exists():
                try:
                    self.data[split] = CustomImageDataset(path, transform=self.transform, split=split)
                    empty = False
                except FileNotFoundError:
                    pass
            if empty:
                print_once(
                    f'Warning: no images found in {path}. Using empty dataset for split {split}. '
                    f'Perhaps you set `dataset.path` incorrectly?')
                self.data[split] = NullIterableDataset(1)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader('test')

    def _get_dataloader(self, split: str):
        return DataLoader(self.data[split], **self.dataloader)
