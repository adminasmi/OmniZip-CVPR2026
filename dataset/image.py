#-- Check Done --#
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bisect
import math
from abc import ABC
import logging

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import transforms


def extract_patches_rgb(image, patch_size):
    """
    Extracts patches of size patch_size from the image array.
    Returns:
        List of patches.
    """
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):     # h, w, 3
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1], :]
            if patch.shape[:2] == tuple(patch_size):
                patches.append(patch)
    return patches


def extract_patches_gray(image, patch_size):
    """
    Extracts patches of size patch_size from the image array.
    Returns:
        List of patches.
    """
    patches = []
    for i in range(0, image.shape[0], patch_size[0]):
        for j in range(0, image.shape[1], patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            if tuple(patch.shape) == tuple(patch_size):
                patches.append(patch)
    return patches

class ImageBaseDataset(Dataset, ABC):
    def __init__(
        self,
        args,
        word2id_dict=None,
        mode='train',
        logger=None
    ):
        super().__init__()
        self.args = args
        self.logger = logger or logging.getLogger('base')
        self.mode = mode

        self.word2id_dict = word2id_dict
        self.start_id = word2id_dict.get('<image>', 0)  # Use modal token instead of <s>
        self.pad_id = word2id_dict.get('<pad>', 0)

        self.patch_size = getattr(args, 'patch_size', (16,16))
        self.padding_unit = getattr(args, 'padding_unit', 16)
        self.chan_corre = getattr(args, 'chan_corre', True)
        self.get_str_pixels = getattr(args, 'get_str_pixels', True)
        self.padding = getattr(args, 'padding', True)
        self.debug = getattr(args, 'debug', False)
        self.num_images = getattr(args, 'num_images', -1)

        if self.mode == 'train':
            self._build_augment()
        self._resolve_paths()

        self.logger.info('Calculating number of patches per image ...')
        self.npatches_per_image = []
        self.cols_per_image = []
        for image_path in self.image_paths:
            self._cal_npatches_per_image(image_path)
        self.cum_counts = np.cumsum(self.npatches_per_image)    # Prefix sum to map global idx to image
        self.npatches = int(self.cum_counts[-1])
        self.logger.info(f'In total have {self.npatches} patches of size {self.patch_size} in {self.image_dir}.')
        
        self.modality = 'image'


    def __len__(self):
        return self.npatches

    def __getitem__(self, index):
        # Locate image index via binary search
        image_idx = bisect.bisect_right(self.cum_counts, index)
        prev_sum = 0 if image_idx == 0 else self.cum_counts[image_idx - 1]
        local_idx = index - prev_sum
        image_path = self.image_paths[image_idx]

        # here use RGB image's logic.
        n_w = self.cols_per_image[image_idx]    # Determine row, col in that image
        row, col = local_idx // n_w, local_idx % n_w
        with Image.open(image_path) as image:
            image = image.convert('L') if image.mode=='L' else image.convert('RGB')
            image = self.transform(image) if (self.mode == 'train' and random.random() < 0.6) else image

            # Padding to multiple of patch size
            ph, pw = self.patch_size
            w,  h  = image.size
            padw, padh = (pw - (w % pw)) % pw, (ph - (h % ph)) % ph
            if padw > 0 or padh > 0:
                image = ImageOps.expand(image, border=(0, 0, padw, padh), fill=0)
            left  = col * self.patch_size[1]    # Crop desired patch
            upper = row * self.patch_size[0]
            right = left + self.patch_size[1]
            lower = upper + self.patch_size[0]
            patch = image.crop((left, upper, right, lower))

        # Tokenize patch
        patch_data = self._tokenize_patch(patch)
        instance = {
            'input_tokens': torch.tensor(patch_data['inputs'], dtype=torch.long),
            'target_tokens': torch.tensor(patch_data['targets'], dtype=torch.long),
            'masks': patch_data['masks'],
            'modality': self.modality,
        }
        return instance


    def _tokenize_patch(self, patch):
        is_gray = (patch.mode=='L')
        patch = np.array(patch)
        if is_gray:
            patch_flat = patch.reshape(-1, 1)
        else:
            patch_flat = patch.reshape(-1, 3) if self.chan_corre else patch.reshape(-1, 3).T

        if self.get_str_pixels:
            patch_tokens = [self.word2id_dict[str(pixel)] for row in patch_flat for pixel in row]
        else:
            patch_tokens = [pixel for row in patch_flat for pixel in row]

        patch_targets = [pixel for row in patch_flat for pixel in row]
        patch_inputs  = [self.start_id] + patch_tokens[:-1]     # shift right for inputs

        seqlen = len(patch_inputs)      # add padding
        pad_len = (self.padding_unit - (seqlen % self.padding_unit)) % self.padding_unit if self.padding else 0
        if pad_len > 0:
            pad_id = self.pad_id if self.get_str_pixels else 0
            patch_inputs += [pad_id] * pad_len

        # create masks
        patch_masks = torch.ones(len(patch_inputs), dtype=torch.bool)
        if pad_len > 0:
            patch_masks[-pad_len:] = False

        if not self.get_str_pixels:
            assert max(patch_inputs) <= 255 and min(patch_inputs) >= 0, f'max target {max(patch_inputs)} exceeds 255 or min target {min(patch_inputs)} belows 0.'
        assert max(patch_targets) <= 255 and min(patch_targets) >= 0, f'max target {max(patch_targets)} exceeds 255 or min target {min(patch_targets)} belows 0.'

        instance = {
            'tokens':  patch_tokens,
            'targets': patch_targets,
            'inputs':  patch_inputs,
            'masks':   patch_masks
        }
        return instance


    def _resolve_paths(self):
        arg_name  = 'image_dir'
        arg_dir   = getattr(self.args, 'image_dir', None)

        if isinstance(arg_dir, list):
            assert len(arg_dir) == 2, f'{arg_name} should contain exactly 2 paths [train, test]'
            self.image_dir = arg_dir[0 if self.mode == 'train' else 1]
        elif isinstance(arg_dir, str):
            self.image_dir = arg_dir
        else:
            raise ValueError(f'Invalid type for {arg_name}: {type(arg_dir)}. Expected str or list.')

        self.image_paths = [
            str(p) for p in Path(self.image_dir).rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.nii')
        ]
        if self.debug and (len(self.image_paths) > 5):
            self.image_paths = self.image_paths[:5]
        elif self.mode == "eval" and (len(self.image_paths) > 50):
            self.image_paths = self.image_paths[:50]
        assert len(self.image_paths) > 0, f'Empty dir in {self.image_dir}.'

    def _build_augment(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def _cal_npatches_per_image(self, image_path):
        # here use RGB image's logic.
        ph, pw = self.patch_size
        with Image.open(image_path) as image:
            image = image.convert('L') if image.mode=='L' else image.convert('RGB')
            W, H = image.size  # PIL: (width, height)
        n_h = math.ceil(H / ph)
        n_w = math.ceil(W / pw)
        self.npatches_per_image.append(n_h * n_w)
        self.cols_per_image.append(n_w)


    def _log_attributes(self):
        self.logger.info(f"========== {self.modality[0].upper() + self.modality[1:].lower()}Dataset Attributes ({self.mode.upper()}) ==========")
        self.logger.info(f"{'get str pixels':<22}: {self.get_str_pixels}")
        self.logger.info(f"{'image_dir':<22}: {self.image_dir}")
        self.logger.info(f"{'image_paths':<22}: {len(self.image_paths)} total")
        if len(self.image_paths) > 3:
            self.logger.info(f"{'':<22}  -> Sample: {self.image_paths[:3]} ...")
        else:
            self.logger.info(f"{'':<22}  -> {self.image_paths}")
        self.logger.info(f"{'patch_size':<22}: {self.patch_size}")
        self.logger.info(f"{'chan_corre':<22}: {self.chan_corre}")
        self.logger.info(f"{'npatches_per_image sample':<22}: {self.npatches_per_image[:3]} ...")
        self.logger.info(f"{'total patches':<22}: {self.npatches}")
        self.logger.info("=============================================\n")


    def _visualize_patches(self, num_patches=16, ncols=4, save_path=None, figsize=(12, 12)):
        """
        Randomly sample and visualize tokenized patches by reconstructing image patches.

        Args:
            num_patches (int): Number of patches to display.
            ncols (int): Number of columns in the grid.
            save_path (str): File path to save the figure. If None, display interactively.
            figsize (tuple): Figure size.
        """
        import matplotlib.pyplot as plt
        import math

        indices = np.random.choice(len(self), num_patches, replace=False)
        nrows = math.ceil(num_patches / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        for i, idx in enumerate(indices):
            ax = axes[i // ncols, i % ncols] if nrows > 1 else axes[i % ncols]
            inst = self[idx]
            pixels = inst['target_tokens'].numpy()
            h, w = self.patch_size
            # Reconstruct array from flat pixels
            if self.chan_corre:
                arr = pixels.reshape(h, w, 3)
            else:
                arr = pixels.reshape(3, h, w).transpose(1, 2, 0)
            ax.imshow(arr.astype(np.uint8))
            ax.axis('off')

        # Turn off empty subplots
        for j in range(i + 1, nrows * ncols):
            ax = axes[j // ncols, j % ncols] if nrows > 1 else axes[j % ncols]
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logging.getLogger('base').info(f"Saved patch grid to {save_path}")
        else:
            plt.show()

class ImageDataset(ImageBaseDataset):
    """ Natural Image Dataset """
    def __init__(
        self, 
        args, 
        word2id_dict=None, 
        mode='train', 
        logger=None
    ):
        super().__init__(args, word2id_dict, mode, logger)
        self.modality = 'image'
        self._log_attributes()
        
        
    


if __name__ == '__main__':
    from utils.common import read_word2id_dict
    from dataset import collate_fn_test
    from dataclasses import dataclass, field
    from typing import List
    from torch.utils.data import DataLoader
    
    @dataclass
    class ARGS:
        data_root: str = '/media/ps/ssd6/zhaoy/datasets'
        image_dir: List[str] = field(default_factory=lambda: ['image/Kodak/Images', 'image/clic/mobile/valid'])
        patch_size: tuple = (16, 16)
        chan_corre: bool = True
        debug: bool = False
    args = ARGS()
    
    dataset = ImageDataset(
        args,
        mode='train',
        word2id_dict=read_word2id_dict('/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json')
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    print(len(dataloader))
    for idx, data in enumerate(dataloader):
        print(data['input_tokens'].shape)
        break