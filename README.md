### OCTET: Object-aware Counterfactual Explanations

## Introduction
This is the official repository for the paper:

[OCTET: Object-aware Counterfactual Explanations](https://arxiv.org/abs/2211.12380), Mehdi Zemni, Mickaël Chen, Éloi Zablocki, Hédi Ben-Younes, Patrick Pérez, Matthieu Cord

OCTET is a counterfactual explanation method for deep visual classifiers.

## Installation and preparation

Clone this repo.
```bash
git clone https://github.com/valeoai/OCTET.git
cd OCTET/
```

This code requires PyTorch (1.8.1), python 3+, and cuda (11.1). Please install dependencies by
```bash
pip install -r requirements.txt
```

All checkpoints are provided in the release, please extract them. It should look like this:
```
OCTET/checkpoints/blobgan_256x512.ckpt
OCTET/checkpoints/decision_densenet.tar
OCTET/checkpoints/drivable
OCTET/checkpoints/encoder_256x512.pt
```

# Usage

To use OCTET, you will need to instantiate a `Config` object with the desired parameters. The `Config` object should contain the following parameters:

`blobgan_weights`: path to the weights file for the blobGan model
`encoder_weights`: path to the weights file for the image encoder
`decision_model_weights`: path to the weights file for the decision model to be explained
`device`: torch device to use for computation
`output_dir`: directory where generated images will be saved
`real_images`: a boolean indicating whether real images should be used (True) or generated images should be used (False)
`dataset_path`: path to the directory containing the images to use (if `real_images` is True)
`bs`: batch size to use when loading images from the dataset
`num_imgs`: number of images to generate (if `real_images` is False)

Once you have created a `Config` object, you can instantiate an OCTET object with `octet = OCTET(config)`. You can then call the following methods on the OCTET object: `octet.invert_and_cf()` that will invert the image to get the latent code and generate the counterfactual explanation.

To run the script with default parameters (although you will need to change data path), simply run
```
python octet_invert_counterfactual.py
```

## Acknowledgements

This code is based on the original [STEEX code](https://github.com/valeoai/STEEX) and [BlobGAN code](https://github.com/dave-epstein/blobgan).


## Disclaimer

There might be some bugs or errors. Feel free to open an issue and/or contribute to improve the repo.
