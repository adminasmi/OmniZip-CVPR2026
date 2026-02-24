import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Match important environment settings from train.py to keep runtime consistent
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

import json
import logging
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch._dynamo
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

# Import arithmetic coding modules
from Evaler import _arithmetic_decode, ARITHMETIC_CODER_PRECISION
from aclibs import arithmetic_coder, bitstreams, frequency_table

def parse_args():
    parser = argparse.ArgumentParser('OmniDecomp', add_help=False)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.'
    )
    parser.add_argument('--text',    action='store_true', help='decompress text only.')
    parser.add_argument('--image',   action='store_true', help='decompress image only.')
    parser.add_argument('--speech',  action='store_true', help='decompress speech only.')
    parser.add_argument('--gene',    action='store_true', help='decompress gene only.')
    parser.add_argument('--tactile', action='store_true', help='decompress tactile only.')
    parser.add_argument('--medical', action='store_true', help='decompress medical only.')
    parser.add_argument('--database', action='store_true', help='decompress database only.')
    
    parser.add_argument('--unify',   action='store_true', help='unified model for all modalities.')
    
    parser.add_argument('--moe',    action='store_true', help='MOE models for different modalities.')
    parser.add_argument('--moa',    action='store_true', help='MOA models for different modalities.')
    
    parser.add_argument('--name', '-n', default='omnidecomp', )
    parser.add_argument('--exp_root', default='./experiments/test', help='path where to save, empty for no saving')

    # tr  aining parameters
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
    
    # strategies
    parser.add_argument('--use_ac',      action='store_true', help='use arithmetic coding')
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
    
    # Decompression specific arguments
    parser.add_argument('--compressed_file', type=str, help='path to compressed file')
    parser.add_argument('--output_file', type=str, help='path to save decompressed output')
    parser.add_argument('--logits_shape', type=int, nargs=3, default=[16, 1024, 16384], help='shape of original logits tensor')
    
    return parser


parser = argparse.ArgumentParser('Decompressing', parents=[parse_args()])
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
            

@torch.no_grad()
def decompress(args):
    '''
    decompress single/multi-modality data.
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
    
    # Check if compressed file exists
    if not args.compressed_file or not os.path.exists(args.compressed_file):
        logger.error(f'Compressed file not found: {args.compressed_file}')
        return
    
    # Load compressed data
    logger.info(f'Loading compressed data from: {args.compressed_file}')
    with open(args.compressed_file, 'rb') as f:
        compressed_data = f.read()
    
    logger.info(f'Compressed data size: {len(compressed_data)} bytes')
    
    # Decompress data
    logger.info('Starting decompression...')
    logits_shape = tuple(args.logits_shape)
    
    try:
        decoded_tokens = _arithmetic_decode(compressed_data, logits_shape, device)
        logger.info(f'Decompression successful. Output shape: {decoded_tokens.shape}')
        
        # Save decompressed data
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = os.path.join(args.exp_root, 'decompressed_tokens.pt')
        
        torch.save(decoded_tokens, output_path)
        logger.info(f'Decompressed data saved to: {output_path}')
        
        # Save as numpy array for easier inspection
        np_output_path = output_path.replace('.pt', '.npy')
        torch.save(decoded_tokens.cpu().numpy(), np_output_path.replace('.pt', '.npy'))
        logger.info(f'Decompressed data also saved as numpy: {np_output_path}')
        
    except Exception as e:
        logger.error(f'Decompression failed: {e}')
        return
    
    logger.info('Decompression completed successfully!')


if __name__ == '__main__':    
    decompress(args)
