#-- Check Done --#
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.image import ImageBaseDataset

class TactileDataset(ImageBaseDataset):
    """ Natural Image Dataset """
    def __init__(
        self, 
        args, 
        word2id_dict=None, 
        mode='train', 
        logger=None
    ):
        super().__init__(args, word2id_dict, mode, logger)
        self.modality = 'tactile'
        # Override start_id with tactile modal token
        self.start_id = word2id_dict.get('<tactile>', 0)
        self._log_attributes()


if __name__ == '__main__':
    from utils.common import read_word2id_dict
    from dataset import collate_fn_test
    from dataclasses import dataclass, field
    from typing import List
    from torch.utils.data import DataLoader
    
    @dataclass
    class ARGS:
        image_dir: List[str] = '/media/ps/ssd6/zhaoy/datasets/touch/TouchandGoDataset-v2/dataset-comp/test/image'
        patch_size: tuple = (16, 16)
        chan_corre: bool = True
        debug: bool = False
    args = ARGS()
    
    dataset = TactileDataset(
        args,
        mode='train',
        word2id_dict=read_word2id_dict('/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json')
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    print(len(dataloader))
    for idx, data in enumerate(dataloader):
        print(data['input_tokens'].shape)
        break