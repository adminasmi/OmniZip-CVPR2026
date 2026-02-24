#-- Check Done --#
import os
import math
import random
import bisect

import torch
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nibabel as nib
from PIL import Image, ImageOps
from dataset.image import ImageBaseDataset

class MedicalDataset(ImageBaseDataset):
    """ Natural Image Dataset """
    def __init__(
        self, 
        args, 
        word2id_dict=None, 
        mode='train', 
        logger=None
    ):
        super().__init__(args, word2id_dict, mode, logger)
        self.modality = 'medical'
        # Override start_id with medical modal token
        self.start_id = word2id_dict.get('<medical>', 0)
        self._log_attributes()
        
        
    def __getitem__(self, index):
        # Locate image index via binary search
        image_idx = bisect.bisect_right(self.cum_counts, index)
        prev_sum = 0 if image_idx == 0 else self.cum_counts[image_idx - 1]
        local_idx = index - prev_sum
        image_path = self.image_paths[image_idx]
        
        if image_path.endswith('.nii'):
            image = nib.load(image_path)
            image = np.array(image.dataobj, dtype=np.uint16)
            n_w = self.cols_per_image[image_idx] // image.shape[2]
            
            # find current slice and local index in that slice
            slice_idx = local_idx // (n_w * (image.shape[0] // self.patch_size[0]))
            local_idx = local_idx %  (n_w * (image.shape[0] // self.patch_size[0]))
            local_idx_in_slice = local_idx % (n_w * (image.shape[0] // self.patch_size[0]))
            
            row, col = local_idx_in_slice // n_w, local_idx_in_slice % n_w
            curr_slice = image[:, :, slice_idx]
            slice_high = (curr_slice >> 8).astype(np.uint8)
            slice_low  = (curr_slice & 0xFF).astype(np.uint8)
            curr_slice = np.concatenate([slice_high, slice_low], axis=-1)
            
            ph, pw = self.patch_size
            upper = row * ph
            left  = col * pw
            patch = curr_slice[upper : upper+ph, left : left+pw]
            patch = Image.fromarray(patch)
            if random.random() < 0.6 and self.mode == 'train':
                patch = self.transform(patch)
        
        else:
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
        
    
    def _cal_npatches_per_image(self, image_path):
        '''Calculate the number of patches in an image based on its size and patch size.'''
        ph, pw = self.patch_size
        if image_path.endswith('.nii'):
            image = nib.load(image_path)
            image = np.array(image.dataobj)
            H, W, n_slices = image.shape[:3]
            n_h = math.ceil(H / ph)
            n_w = math.ceil(W / pw)
            self.npatches_per_image.append(n_h * n_w * n_slices)
            self.cols_per_image.append(n_w * n_slices)
        else:
            with Image.open(image_path) as image:
                image = image.convert('L') if image.mode=='L' else image.convert('RGB')
                W, H = image.size  # PIL: (width, height)
            n_h = math.ceil(H / ph)
            n_w = math.ceil(W / pw)
            self.npatches_per_image.append(n_h * n_w)
            self.cols_per_image.append(n_w)


if __name__ == '__main__':
    from utils.common import read_word2id_dict
    from dataset import collate_fn_test
    from dataclasses import dataclass, field
    from typing import List
    from torch.utils.data import DataLoader
    
    @dataclass
    class ARGS:
        image_dir: str = '/media/ps/ssd6/zhaoy/datasets/medical/MosMedData/studies/CT-4'
        patch_size: tuple = (16, 16)
        chan_corre: bool = True
        debug: bool = False
    args = ARGS()
    
    dataset = MedicalDataset(
        args,
        mode='train',
        word2id_dict=read_word2id_dict('/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json')
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    print(len(dataloader))
    for idx, data in enumerate(dataloader):
        print(data['input_tokens'].shape)
        break