import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Match important environment settings from train.py to keep runtime consistent
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

from Evaler import evaluate
import warnings
warnings.filterwarnings('ignore')

import json
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch._dynamo
from torch.utils.data import DataLoader
torch._dynamo.config.suppress_errors = True

# follow train.py settings for precision and cudnn
torch.set_float32_matmul_precision('high')

from utils import DictAction, log_args
from utils.common import get_timestamp, setup_logger, read_word2id_dict, clean_state_dict

import warnings
warnings.filterwarnings('ignore')

UNK_TYPE    = 'unk_allow'
VOCAB_SIZE = 16384
COVERAGE    = 1.0
DATA_ROOT   = '/data/ssd/zhaoy/datasets'
CORPUS_ROOT = f'./corpus/{UNK_TYPE}'

def parse_args():
    parser = argparse.ArgumentParser('OmniComp', add_help=False)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
    )
    parser.add_argument('--text',    action='store_true', help='compress text only.')
    parser.add_argument('--image',   action='store_true', help='compress image only.')
    parser.add_argument('--speech',  action='store_true', help='compress speech only.')
    parser.add_argument('--gene',  action='store_true', help='compress speech only.')
    parser.add_argument('--tactile',    action='store_true', help='compress tactile only.')
    parser.add_argument('--medical',   action='store_true', help='compress medical only.')
    parser.add_argument('--database',  action='store_true', help='compress database only.')
    
    parser.add_argument('--unify',   action='store_true', help='unified model for all modalities.')
    
    parser.add_argument('--moe',    action='store_true', help='MOE models for different modalities.')
    parser.add_argument('--moa',    action='store_true', help='MOA models for different modalities.')
    
    parser.add_argument('--name', '-n', default='omnicomp', )
    parser.add_argument('--exp_root', default='./experiments/test', help='path where to save, empty for no saving')

    # training parameters
    parser.add_argument('--pretrain_model', help='load from other checkpoint', default='./checkpoints/dualcomp/dualcomp-300k.pth')
    parser.add_argument('--batch_size', type=int, default=16)
    
    parser.add_argument('--spm_model',  type=str, default=f'./vocabs/{UNK_TYPE}/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_{VOCAB_SIZE}_{COVERAGE}.model')
    parser.add_argument('--vocab_dict', type=str, default=f'./vocabs/{UNK_TYPE}/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_{VOCAB_SIZE}_{COVERAGE}.json')
    
    # image
    parser.add_argument('--num_images', type=int, default=-1, help='number of training images.')
    parser.add_argument('--chan_corre', type=bool, default=True)
    parser.add_argument('--patch_size', type=int, nargs=2, default=(16, 16), help='Patch size as (height, width)')
    parser.add_argument('--narrow_vocab', type=str, default=f'./vocabs/{UNK_TYPE}/vocab_spm_bpe_{VOCAB_SIZE}_{COVERAGE}/spm_bpe_byte_uint8_255.model')
    
    # text
    parser.add_argument('--seq_len', type=int, default=1024)    # used for both text and speech
    
    # speech
    parser.add_argument('--num_speeches', type=int, default=5000, help='number of speech training samples.')
    
    # strategirs
    parser.add_argument('--use_ac',      action='store_true')
    parser.add_argument('--debug',       action='store_true')
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--gpu_ids',     default=['6'], nargs='+', help='device to use for training / testing')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default='rwkv7_hira_switch_moe', help='model name')
    parser.add_argument('--model_size', type=str, default='xs', help='model size')

    parser.add_argument('--mlp_factor',  type=int, default=4, help='hidden factor for MLPs (divide by 2 is the real value)')
    parser.add_argument('--num_experts', type=int, default=4, help='number of experts')
    parser.add_argument('--k',           type=int, default=2, help='number of experts to use')
    parser.add_argument('--num_moe_layers', type=int, default=2, help='number of layers to use MoE')
    parser.add_argument('--num_moa_layers', type=int, default=2, help='number of layers to use MoA')
    parser.add_argument('--hira_factor',   type=int, help='factor of high-rank reparam', default=4)
    return parser


parser = argparse.ArgumentParser('Evaluating', parents=[parse_args()])
args   = parser.parse_args()

gpus = args.gpu_ids if isinstance(args.gpu_ids, list) else [args.gpu_ids]
gpus = ','.join(str(x) for x in gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    

from Trainer import MODALITIES, NUM_WORKERS
if args.unify:
    args.modalities = MODALITIES
else:
    args.modalities = []
    for m in MODALITIES:
        if getattr(args, m, False):
            args.modalities.append(m)
            
from models.build import build_model
from dataset.build import build_task_datasets
from dataset import collate_fn_test


@torch.no_grad()
def compress(args):
    '''
    compress single/multi-modality data.
    '''
    debug_label = '_debug_' if args.debug else '_'
    args.exp_root = os.path.join(args.exp_root, f'{args.name}{debug_label}{get_timestamp()}')
    Path(args.exp_root).mkdir(parents=True, exist_ok=True)
    
    # logger and deal configs
    setup_logger(logger_name='base', dir=args.exp_root, phase=args.name, level=logging.INFO, to_screen=True, to_file=True)
    logger = logging.getLogger('base')
    with open(os.path.join(args.exp_root, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    args.inference = True
    log_args(args, logger)
    
    word2id_dict    = read_word2id_dict(args.vocab_dict)
    args.vocab_size = len(word2id_dict)
    
    args.multi_modality = len(args.modalities) > 0
    
    model, _ = build_model(args)
    model = model.eval()
    model.count_params()
    
    
    # load params
    assert (args.pretrain_model is not None) and os.path.exists(args.pretrain_model) and os.path.isfile(args.pretrain_model)
    try:
        checkpoint = torch.load(args.pretrain_model, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = OrderedDict({k:v for k,v in clean_state_dict(checkpoint['model']).items()})
        else:
            state_dict = OrderedDict({k:v for k,v in clean_state_dict(checkpoint).items()})
        load_info  = model.load_state_dict(state_dict, strict=True)
        logger.info(str(load_info) + '\n\n')
    except Exception as e:
        logger.info(f'Error in loading checkpoints: {e}.')
        return
    
    if 'hira' in args.model_name:
        model.switch_to_inference()
        logger.info(f'switch to inference mode.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)    # NOTE: must after `switch_to_inference()`
    save_stats_dir = os.path.join(args.exp_root, 'expert_analysis')
    
    # build datasets dynamically
    logger.info('\n================ Building Test Sets ================')
    test_sets = build_task_datasets(args, word2id_dict, mode='test', modalities=args.modalities)
    logger.info('===================================================\n')
    
    # evaluate each modality automatically
    for mod, _mod_test_set in test_sets.items():
        if _mod_test_set is None:
            continue
        
        if isinstance(_mod_test_set, dict):    # multiple datasets
            for ds_name, ds in _mod_test_set.items():
                dataloader = DataLoader(
                    ds, shuffle=False, drop_last=False, batch_size=args.batch_size, pin_memory=False, collate_fn=collate_fn_test, num_workers=NUM_WORKERS
                )
                logger.info(f'\n>>> Evaluating [{mod}] / [{ds_name}] ({len(ds)} samples)')
                evaluate(args, mod, ds, dataloader, model, device, logger=logger, save_stats_dir=save_stats_dir)
        else:
            dataloader = DataLoader(
                _mod_test_set, shuffle=False, drop_last=False, batch_size=args.batch_size, pin_memory=False, collate_fn=collate_fn_test, num_workers=NUM_WORKERS
            )
            logger.info(f'\n>>> Evaluating [{mod}] / ({len(_mod_test_set)} samples)')
            evaluate(args, mod, _mod_test_set, dataloader, model, device, logger=logger, save_stats_dir=save_stats_dir)


if __name__ == '__main__':    
    compress(args)