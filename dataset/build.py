# config datasets for training.
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import ConcatDataset
from image import ImageDataset
from medical import MedicalDataset
from tactile import TactileDataset
from text import TextDataset
from database import SQLDataset
from gene import GeneDataset
from speech import SpeechDataset


UNK_TYPE    = 'unk_allow'
VOCAB_SIZE  = 16384
COVERAGE    = 1.0
DATA_ROOT   = '/media/ps/ssd6/zhaoy/datasets'
CORPUS_ROOT = f'/home/zhaoy/OmniZip-CVPR2026/corpus/{UNK_TYPE}'


DATASETS = {
    # ===== Image =====
    'kodak': {
        'mod': 'image', 
        'train': None, 
        'test': f'{DATA_ROOT}/image/Kodak/Images'
    },
    'clic-pro': {
        'mod': 'image', 
        'train': f'{DATA_ROOT}/image/clic/professional/train', 
        'test': f'{DATA_ROOT}/image/clic/professional/valid'
    },
    'clic-mobile': {
        'mod': 'image', 
        'train': f'{DATA_ROOT}/image/clic/mobile/train', 
        'test': f'{DATA_ROOT}/image/clic/mobile/valid'
    },
    'div2k': {
        'mod': 'image', 
        'train': f'{DATA_ROOT}/image/DIV2K/train', 
        'test': f'{DATA_ROOT}/image/DIV2K/valid'
    },

    # ===== Medical =====
    'axial': {
        'mod': 'medical', 
        'train': f'{DATA_ROOT}/medical/MRNet-v1.0/train/axial', 
        'test': f'{DATA_ROOT}/medical/MRNet-v1.0/valid/axial'
    },
    'coronal': {
        'mod': 'medical', 
        'train': f'{DATA_ROOT}/medical/MRNet-v1.0/train/coronal', 
        'test': f'{DATA_ROOT}/medical/MRNet-v1.0/valid/coronal'
    },
    'sagittal': {
        'mod': 'medical', 
        'train': f'{DATA_ROOT}/medical/MRNet-v1.0/train/sagittal', 
        'test': f'{DATA_ROOT}/medical/MRNet-v1.0/valid/sagittal'
    },
    'mosmeddata': {
        'mod': 'medical', 
        'train': f'{DATA_ROOT}/medical/MosMedData/studies/CT-2', 
        'test': f'{DATA_ROOT}/medical/MosMedData/studies/CT-3'
    },

    # ===== Tactile =====
    'touchandgo': {
        'mod': 'tactile', 
        'train': f'{DATA_ROOT}/touch/TouchandGoDataset-v2/dataset-comp/train/image', 
        'test': f'{DATA_ROOT}/touch/TouchandGoDataset-v2/dataset-comp/test/image'
    },
    'objectfolder': {
        'mod': 'tactile', 
        'train': f'{DATA_ROOT}/touch/ObjectFolder_1.0/dataset-comp/train/image', 
        'test': f'{DATA_ROOT}/touch/ObjectFolder_1.0/dataset-comp/test/image'
    },

    # ===== Text =====
    'enwik8': {
        'mod': 'text', 
        'train': {'raw': f'{DATA_ROOT}/text/enwik8', 'corpus': f'{CORPUS_ROOT}/spm_enwik8_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': None
    },
    'enwik9': {
        'mod': 'text', 
        'train': None, 
        'test': {'raw': f'{DATA_ROOT}/text/enwik9', 'corpus': f'{CORPUS_ROOT}/spm_enwik9_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },
    'gutenberg': {
        'mod': 'text', 
        'train': {'raw': f'{DATA_ROOT}/text/gutenberg_train', 'corpus': f'{CORPUS_ROOT}/spm_gutenberg_train_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': {'raw': f'{DATA_ROOT}/text/gutenberg', 'corpus': f'{CORPUS_ROOT}/spm_gutenberg_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },

    # ===== Gene =====
    'genoseq': {
        'mod': 'gene', 
        'train': {'raw': f'{DATA_ROOT}/gene/GenoSeq/GenoSeq-train.fasta', 'corpus': f'{CORPUS_ROOT}/spm_genoseq_train_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': {'raw': f'{DATA_ROOT}/gene/GenoSeq/GenoSeq-test.fasta', 'corpus': f'{CORPUS_ROOT}/spm_genoseq_test_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },
    'dnacorpus': {
        'mod': 'gene', 
        'train': {'raw': f'{DATA_ROOT}/gene/DNACorpus', 'corpus': f'{CORPUS_ROOT}/spm_dnacorpus_train_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': {'raw': f'{DATA_ROOT}/gene/DNACorpus', 'corpus': f'{CORPUS_ROOT}/spm_dnacorpus_test_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },

    # ===== Database =====
    'wikisql': {
        'mod': 'database', 
        'train': {'raw': f'{DATA_ROOT}/database/WikiSQL/train.sql', 'corpus': f'{CORPUS_ROOT}/spm_wikisql_train_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': {'raw': f'{DATA_ROOT}/database/WikiSQL/test.sql', 'corpus': f'{CORPUS_ROOT}/spm_wikisql_test_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },
    'spider': {
        'mod': 'database', 
        'train': {'raw': f'{DATA_ROOT}/database/spider/train', 'corpus': f'{CORPUS_ROOT}/spm_spider_train_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}, 
        'test': {'raw': f'{DATA_ROOT}/database/spider/test', 'corpus': f'{CORPUS_ROOT}/spm_spider_test_bpe_{VOCAB_SIZE}_{COVERAGE}.txt'}
    },

    # ===== Speech =====
    'librispeech': {
        'mod': 'speech', 
        'train': f'{DATA_ROOT}/speech/LibriSpeech/conv2text_uint8/LibriSpeech-train-clean', 
        'test': f'{DATA_ROOT}/speech/LibriSpeech/conv2text_uint8/LibriSpeech-test-clean'
    },
}


TYPE_TO_BUILD_FUNC = {
    'image':     ImageDataset,
    'medical':   MedicalDataset,
    'tactile':   TactileDataset,
    'text':      TextDataset,
    'gene':      GeneDataset,
    'database':  SQLDataset,
    'speech':    SpeechDataset,
}

TASKS_TRAIN = {
    'image':     ['div2k', 'clic-pro', 'clic-mobile'],
    'medical':   ['axial', 'coronal', 'sagittal'],
    'tactile':   ['objectfolder', 'touchandgo'],
    'text':      ['enwik8', 'gutenberg'],
    'speech':    ['librispeech'],
    'gene':      ['genoseq', 'dnacorpus'],
    'database':  ['spider', 'wikisql'],
}

TASKS_EVAL = {
    'image':     ['kodak'],
    'medical':   ['axial'],
    'tactile':   ['objectfolder'],
    'text':      ['gutenberg'],
    'speech':    ['librispeech'],
    'gene':      ['dnacorpus'],
    'database':  ['wikisql'],
}

TASKS_TEST = {
    'image':     ['div2k', 'kodak', 'clic-pro', 'clic-mobile'],
    'medical':   ['axial', 'coronal', 'sagittal'],
    'tactile':   ['objectfolder', 'touchandgo'],
    'text':      ['enwik9', 'gutenberg'],
    'speech':    ['librispeech'],
    'gene':      ['genoseq', 'dnacorpus'],
    'database':  ['spider', 'wikisql'],
}


def _build_single_dataset(args, word2id_dict, name, mode, **kwargs):
    cfg = DATASETS[name]
    mod = cfg['mod']
    path  = cfg['train'] if mode == 'train' else cfg['test']
    if not path:
        return None, None, None

    idx = 0 if mode == 'train' else 1
    if mod in ['image', 'medical', 'tactile']: 
        args.image_dir = [None, None]
        args.image_dir[idx] = path
    elif mod == 'speech':
        args.speech_dir = [None, None]
        args.speech_dir[idx] = path
    elif mod in ['text', 'gene', 'database']:
        if isinstance(path, dict):
            args.text_corpus = path['corpus']
            args.text_file   = path['raw']
        else:
            raise NotImplementedError(f"Unexpected path mod for {mod}: {path}")

    print(f'- Building [{mod}] dataset: {name} ({mode}) @ {path}')

    build_func = TYPE_TO_BUILD_FUNC[mod]
    dataset = build_func(args, mode=mode, word2id_dict=word2id_dict, **kwargs)
    return dataset, mod, name


def _build_datasets(args, word2id_dict, names, mode, task_keys, **kwargs):
    """直接按 mod 聚合"""
    if mode == 'train':
        buckets = {mod: [] for mod in task_keys}
        for name in names:
            ds, mod, ds_name = _build_single_dataset(args, word2id_dict, name, mode, **kwargs)
            if ds is not None:
                buckets[mod].append(ds)

        results = {}
        for mod, lst in buckets.items():
            if len(lst) == 0:
                results[mod] = None
            elif len(lst) == 1:
                results[mod] = lst[0]
            else:
                results[mod] = ConcatDataset(lst)
        return results

    else:
        buckets = {mod: {} for mod in task_keys}
        for name in names:
            ds, mod, ds_name = _build_single_dataset(args, word2id_dict, name, mode, **kwargs)
            if ds is not None:
                buckets[mod][ds_name] = ds
        return buckets


def build_task_datasets(args, word2id_dict, mode, modalities=None, **kwargs):
    """
    Build datasets for specified modalities.
    Args:
        modalities: list[str], e.g. ['image', 'text', 'speech', 'tactile', 'gene', 'database', 'speech']
    """
    if mode == 'train':
        task_dict = TASKS_TRAIN
    elif mode == 'eval':
        task_dict = TASKS_EVAL
    elif mode == 'test':
        task_dict = TASKS_TEST
    else:
        raise ValueError
    
    if modalities is None:
        modalities = list(task_dict.keys())
        

    names = []
    for mod in modalities:
        if mod not in task_dict:
            continue
        names.extend(task_dict[mod])
    return _build_datasets(args, word2id_dict, names, mode, task_keys=task_dict.keys(), **kwargs)


if __name__ == '__main__':
    from dataclasses import dataclass
    from utils.common import read_word2id_dict

    @dataclass
    class ARGS:
        debug: bool = False
        seq_len: int = 1024
        patch_size: tuple = (16, 16)
        chan_corre: bool = True
        vocab_dict: str = '/home/zhaoy/OmniZip-CVPR2026/vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json'

    args = ARGS()
    word2id_dict = read_word2id_dict(
        f'/home/zhaoy/OmniZip-CVPR2026/vocabs/{UNK_TYPE}/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_{VOCAB_SIZE}_{COVERAGE}.json'
    )

    results = build_task_datasets(args, word2id_dict, mode='eval')
    print("All datasets:", results.keys())

