import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore')

import json
import logging
import argparse
from pathlib import Path
import pandas as pd

from Evaler import evaluate
from utils import DictAction, log_args
from utils.common import get_timestamp, setup_logger

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
    parser.add_argument('--exp_root', default='./experiments/train', help='path where to save, empty for no saving')

    # training parameters
    parser.add_argument('--pretrain_model', help='load from other checkpoint', default=None)
    parser.add_argument('--resume',  action='store_true')
    parser.add_argument('--nepochs', type=int, default=20)
    parser.add_argument('--nsteps',  type=int, default=20000, help='max number of iterations in each epoch.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_interval', type=int, default=10000)
    
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
    
    # Batch composition for dual-modal training
    parser.add_argument('--image_ratio',   type=float, default=0.33, help='Ratio of images in each batch (between 0 and 1)')
    parser.add_argument('--speech_ratio',  type=float, default=0.33, help='Ratio of speeches in each batch (between 0 and 1)')
    
    # strategirs
    parser.add_argument('--use_ac',      action='store_true')
    parser.add_argument('--debug',       action='store_true')
    parser.add_argument('--amp',         action='store_true')
    parser.add_argument('--accupdate',   action='store_true')
    parser.add_argument('--accum_steps', type=int, default=4)
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

    # training
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.99), help='betas for FusedAdam')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for FusedAdam')
    parser.add_argument('--max_norm', type=float, default=5.0, help='weight norm clipping')

    # scheduler
    parser.add_argument('--scheduler', type=str, default='cosine_annealing')
    parser.add_argument('--lr', type=float, default=1.2e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--T_mult', type=int, default=4)
    parser.add_argument('--eta_min', type=float, default=1e-5)   
    
    return parser

parser = argparse.ArgumentParser('Training', parents=[parse_args()])
args = parser.parse_args()

gpus = args.gpu_ids if isinstance(args.gpu_ids, list) else [args.gpu_ids]
gpus = ','.join(str(x) for x in gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpus

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

from Trainer import Trainer, MODALITIES
if args.unify:
    args.modalities = MODALITIES
else:
    args.modalities = []
    for m in MODALITIES:
        if getattr(args, m, False):
            args.modalities.append(m)
    
from models.build import save_ckpts, manage_ckpts
from utils import BestMetricHolder


def eval_period(args, trainer, stage, epoch, rows, logger):
    save_stats_dir = os.path.join(args.exp_root, 'expert_analysis', f'stage{stage}_epoch{epoch}')
    os.makedirs(save_stats_dir, exist_ok=True)
    
    results = {}
    total_bpb, total_cnt = 0.0, 0
    for mod, entries in trainer.eval_registry.items():
        results[mod] = {}
        
        for entry in entries:
            logger.info(f'\n### Evaluating [{mod}] / [{entry.name}] ###')
            bpb = evaluate(
                args, modality=entry.modality, dataset=entry.dataset, dataloader=entry.loader, model=trainer.model, device=trainer.device, logger=logger, save_stats_dir=os.path.join(save_stats_dir, mod, entry.name)
            )
            results[mod][entry.name] = bpb
            total_bpb += bpb
            total_cnt += 1
            if (args.moe or args.moa) and hasattr(trainer.model, 'reset_expert_stats'):
                trainer.model.reset_expert_stats()
            logger.info(f'==> Done: {entry.name} (bpb={bpb:.4f})\n')

    row = {'epoch': epoch}
    for mod, ds_bpb in results.items():
        for name, bpb in ds_bpb.items():
            row[f'{mod}/{name}'] = bpb
    rows.append(row)
    
    avg_bpb = total_bpb / total_cnt if total_cnt > 0 else 0.0
    logger.info(f'\n>>> Evaluation finished: Avg bpb={avg_bpb:.4f} across {total_cnt} datasets\n')
    return avg_bpb
    

def train(args):
    '''
    Trains a transformer decoder model on Enwik8 data with periodic evaluation.

    Args:
        dataset_train: Training dataset.
        dataset_test: Validation dataset.
        model_size: Size identifier for the model configuration.
        nepochs: Number of training epochs.
        log_every: Frequency of logging the loss (in steps).
        eval_every: Frequency of evaluation (in epochs).
        batch_size: Number of sequences per batch.
        chunk_size: Length of each sequence.
        use_tqdm_flag: Whether to display a progress bar.
        debug: If True, run only a few steps for debugging.
        save_dir: Directory to save model checkpoints.
        model_name: Base name for saved model files.
        device: Device to run the training on.

    Returns:
        A tuple containing the best model state_dict and the best validation loss.
    '''
    # Setup experiment directory
    debug_label = '_debug_' if args.debug else '_'
    args.exp_root = os.path.join(args.exp_root, f'{args.name}{debug_label}{get_timestamp()}')
    Path(args.exp_root).mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = os.path.join(args.exp_root, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Set up logger
    setup_logger(logger_name='base', dir=args.exp_root, phase=args.name, level=logging.INFO, to_screen=True, to_file=True)
    logger = logging.getLogger('base')
    
    with open(os.path.join(args.exp_root, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    args.inference = False
    log_args(args, logger)
    
    # Create trainer
    trainer = Trainer(args, logger=logger)
    metricholder = BestMetricHolder(init_value=100.0, better='smaller', use_ema=False)
    
    rows = []
    columns = ['epoch']
    if trainer.multi_modality:
        columns.extend(args.modalities)
    else:
        for m in MODALITIES:
            if getattr(args, m, False):
                columns.append(m)

    # Two-stage training
    if trainer.multi_modality and args.moa and args.moe:
        stages = list(range(trainer.init_stage, 3)) if trainer.init_stage < 2 else [2]
    else:
        stages = [2]
    for stage in stages:
        logger.info(f'\n\nStarting training stage {stage}...')
        
        # Set parameter freezing states for current stage
        trainer._set_training_stage(stage)
        trainer._set_optimizer(args, stage)
        trainer._set_scheduler(args, stage)
        trainer._load_training_state(args, only_progress=False)
        
        # Training loop for current stage
        nepochs = 1 if stage in [0, 1] else args.nepochs
        for epoch in range(trainer.init_epoch, trainer.init_epoch + nepochs):
            args.inference = False
            
            logger.info(f'\n\nTraining epoch {epoch} ...')
            trainer.train_single_epoch_multi_modality(
                args, stage, epoch, ckpt_dir, 
            )
            manage_ckpts(ckpt_dir, stage, epoch, trainer, args.model_name, model_size=args.model_size)
            
            if args.debug or (stage > 1 and epoch % 2 == 0):
                # validate if it is the best
                args.inference = True
                
                avg_bpb = eval_period(args, trainer, stage, epoch, rows, logger)
                isbest = metricholder.update(avg_bpb, epoch, is_ema=False)
                if isbest:
                    ckpt_path = os.path.join(ckpt_dir, f'{args.model_name}_{args.model_size}_stage{stage}_best.pth')
                    save_ckpts(ckpt_path, stage, epoch, trainer)
                    logger.info(f'Saving best checkpoint to {ckpt_path}.\n\n')

            if args.debug and epoch >= 2:
                logger.info('Debug. Break!')
                break
    
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(os.path.join(args.exp_root, 'eval.csv'), index=False)
            
            
if __name__ == '__main__':    
    train(args)