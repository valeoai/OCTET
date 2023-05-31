import os
from typing import Dict, Tuple, List
from typing import Optional, Callable, Any

from PIL import Image
import torch
from torchvision.datasets.folder import default_loader, ImageFolder, make_dataset
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


import json
import numpy as np
from torchvision import transforms as TR

from utils import is_rank_zero, print_once

EXT = ('.jpg', '.png')
LABELS = np.array([
'Forward - follow traffic',
'Forward - the road is clear',
'Forward - the traffic light is green',
'Stop/slow down - obstacle: car',
'Stop/slow down - obstacle: person/pedestrain',
'Stop/slow down - obstacle: rider',
'Stop/slow down - obstacle: others',
'Stop/slow down - the traffic light',
'Stop/slow down - the traffic sign',
'Turn left - front car turning left',
'Turn left - on the left-turn lane',
'Turn left - traffic light allows',
'Turn right - front car turning right',
'Turn right - on the right-turn lane',
'Turn right - traffic light allows',
'Can not turn left - obstacles on the left lane',
'Can not turn left - no lane on the left',
'Can not turn left - solid line on the left',
'Can not turn right - obstacles on the right lane',
'Can not turn right - no lane on the right',
'Can not turn right - solid line on the left'])

class ImageFolderWithFilenames(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None):
        super().__init__(root=root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)

    @staticmethod
    def make_dataset(
            directory: str,
            class_to_idx: Dict[str, int],
            extensions: Optional[Tuple[str, ...]] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError(
                "The class_to_idx parameter cannot be None."
            )
        cache_path = os.path.join(directory, 'cache.pt')
        try:
            dataset = torch.load(cache_path, map_location='cpu')
            print_once(f'Loading dataset from cache in {directory}')
        except FileNotFoundError:
            print_once(f'Creating dataset and saving to cache in {directory}')
            dataset = make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
            if is_rank_zero():
                torch.save(dataset, cache_path)
        return dataset

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return x, {'labels': y, 'filenames': self.imgs[i][0]}

class CustomImageDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None, split='train'):
        self.root = root
        self.transform = transform
        self.imgs = os.listdir(root)
        self.imgs = sorted([p for p in self.imgs if p.endswith(EXT)])#*300
        #if split == 'train':
        #    with open('sunny.txt') as f:
        #        lines = f.read().splitlines()
        #    self.imgs = list(set(self.imgs) & set(lines))


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img_path = os.path.join(self.root, self.imgs[i])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.imgs[i]


class BDDOIADataset(Dataset):
    def __init__(self, imageRoot, gtRoot, reasonRoot, cropSize=(1280, 720), augment=False):

        super(BDDOIADataset, self).__init__()

        self.imageRoot = imageRoot
        self.gtRoot = gtRoot
        self.reasonRoot = reasonRoot
        self.cropSize = cropSize
        self.augment = augment

        with open(gtRoot) as json_file:
            data = json.load(json_file)
        with open(reasonRoot) as json_file:
            reason = json.load(json_file)

        data['images'] = sorted(data['images'], key=lambda k: k['file_name'])
        reason = sorted(reason, key=lambda k: k['file_name'])

        # get image names and labels
        action_annotations = data['annotations']
        imgNames = data['images']
        self.imgNames, self.targets, self.reasons = [], [], []
        for i, img in enumerate(imgNames):
            ind = img['id']
            if len(action_annotations[ind]['category']) == 4  or action_annotations[ind]['category'][4] == 0:
                file_name = os.path.join(self.imageRoot, img['file_name'])
                if os.path.isfile(file_name):
                    self.imgNames.append(file_name)
                    self.targets.append(torch.LongTensor(action_annotations[ind]['category']))
                    self.reasons.append(torch.LongTensor(reason[i]['reason']))

        self.count = len(self.imgNames)
        print("number of samples in dataset:{}".format(self.count))

    def __len__(self):
        return self.count

    def __getitem__(self, ind):
        imgName = self.imgNames[ind]

        raw_image = Image.open(imgName).convert('RGB')
        target = np.array(self.targets[ind], dtype=np.int64)
        reason = np.array(self.reasons[ind], dtype=np.int64)

        image, target, reason = self.transforms(raw_image, target, reason)

        return {"image": image, "target":target, "reason":reason, "name":imgName}

    def transforms(self, raw_image, target, reason):

        if self.augment:
            pass

        new_width, new_height = (self.cropSize[1], self.cropSize[0])

        image = TR.functional.resize(raw_image, 256, InterpolationMode.LANCZOS)
        image = TR.functional.center_crop(image, 256)

        image = TR.functional.to_tensor(image)
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        target = torch.FloatTensor(target)[0:4]
        reason = torch.FloatTensor(reason)

        return image, target, reason
