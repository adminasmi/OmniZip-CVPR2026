import torch
from rwkv7_hira_moe import *
from rwkv7_hira_moa import *

from rwkv7_hira_kmoa_moe import *
from rwkv7_hira_rmoa_moe import *
from rwkv7_hira_vmoa_moe import *
from rwkv7_hira_kvmoa_moe import *
from rwkv7_hira_rkmoa_moe import *
from rwkv7_hira_rvmoa_moe import *
from rwkv7_hira_rkvmoa_moe import *

from transformer import *
from mamba import *

from glob import glob
torch.set_float32_matmul_precision('high')

RWKV_MODEL_CONFIGS = {
        'xs':  dict(n_embd=96,  n_layer=2),     # 0.4m
        's':   dict(n_embd=320, n_layer=2),     # 4.8m
        'm':   dict(n_embd=912, n_layer=2),     # 38.0m
        'l':   dict(n_embd=1488, n_layer=3),    # 152.0m
        # 'xl':  dict(n_embd=1184, n_layer=3),
        # 'xxl': dict(n_embd=1456, n_layer=4)
    }

MAMBA_MODEL_CONFIGS = {
        'xs':  dict(n_embd=88,  n_layer=2),     # 0.2m
        's':   dict(n_embd=356, n_layer=2),     # 3.2m
    }

TSFM_MODEL_CONFIGS = {
        'xs':  dict(n_embd=96,  n_layer=2),     # 0.2m
        's':   dict(n_embd=368, n_layer=2),     # 3.2m
    }



def build_model(args, **kwargs):
    """ use registery to manage models. """
    assert args.model_name in MODULE_BUILD_FUNCS._module_dict, f"Model {args.model_name} not found in registry."
    build_func = MODULE_BUILD_FUNCS.get(
        args.model_name
    )
    if args.model_name == 'mamba':
        model_configs = MAMBA_MODEL_CONFIGS
    elif args.model_name == 'transformer':
        model_configs = TSFM_MODEL_CONFIGS
    else:
        model_configs = RWKV_MODEL_CONFIGS
    args.n_embd  = model_configs[args.model_size]['n_embd']
    args.n_layer = model_configs[args.model_size]['n_layer']
    model, criterion = build_func(
        args, **kwargs
    )
    return model, criterion


def save_ckpts(ckpt_path, stage, epoch, trainer):
    """Save model checkpoint."""
    state_dict = {
        'model': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
        'stage': stage,
        'epoch': epoch,
    }
    if trainer.scheduler is not None:
        state_dict['scheduler'] = trainer.scheduler.state_dict()
    torch.save(state_dict, ckpt_path)


def manage_ckpts(ckpt_dir, stage, epoch, trainer, model_name, model_size, save_interval=1):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_files = [
        ckpt for ckpt in glob(os.path.join(ckpt_dir, '*.pth'))
        if os.path.basename(ckpt) not in {f'{model_name}.pth', f'{model_name}_best.pth'}
    ]
    ckpt_files.sort()

    # remove the oldest checkpoints if there are 3 or more.
    if len(ckpt_files) > 3:
        os.remove(ckpt_files[0])

    # save new checkpoint at specified intervals
    if (epoch + 1) % save_interval == 0:
        ckpt_path = os.path.join(ckpt_dir, f'{model_name}_{model_size}.pth')
        save_ckpts(ckpt_path, stage, epoch, trainer)
        