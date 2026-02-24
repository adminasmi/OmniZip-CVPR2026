#-- Check Done --#
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.text import TextBaseDataset

class GeneDataset(TextBaseDataset):
    def __init__(
        self, args, word2id_dict, mode='train', logger=None
    ):
        super().__init__(args, word2id_dict, mode, logger)
        self.modality = 'gene'
        # Override start_id with gene modal token
        self.start_id = word2id_dict.get('<gene>', 0)
        self._log_attributes()
        
        
if __name__ == '__main__':
    from dataclasses import dataclass
    from dataset import collate_fn_test
    from utils.common import read_word2id_dict
    from torch.utils.data import DataLoader
    
    UNK_TYPE    = 'unk_allow'
    VOCAB_SIZE  = 16384
    COVERAGE    = 1.0
    
    @dataclass
    class ARGS:
        debug : bool = True
        seq_len : int = 1024
        text_file: str = '/media/ps/ssd6/zhaoy/datasets/Gene/DNACorpus/test/BuEb'
        text_corpus: str = f'./corpus/{UNK_TYPE}/spm_dnacorpus_test_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'
    args = ARGS()
    
    dataset = GeneDataset(
        args=args,
        mode='train',
        word2id_dict=read_word2id_dict(f'/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_{VOCAB_SIZE}_{COVERAGE}.json')
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, collate_fn=collate_fn_test)
    for idx, data in enumerate(dataloader):
        print(data['input_tokens'])
        print(data['input_tokens'].shape)
        break