import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import inspect
import torch
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
torch.set_float32_matmul_precision('high')

import logging
from dataclasses import dataclass
from collections import OrderedDict

from models.build import build_model, save_ckpts
from utils.common import read_word2id_dict, clean_state_dict
from vocabs.narrvocab import sample_logits
from dataset import collate_fn_train, collate_fn_test, BalancedBatchSampler
from dataset.build import build_task_datasets

import warnings
warnings.filterwarnings('ignore')


NUM_WORKERS = 12
SCHEDULERS = {
    'multi_epoch': lr_sched.MultiStepLR,
    'step_lr':     lr_sched.StepLR,
    'exponential_lr':   lr_sched.ExponentialLR,
    'cosine_annealing': lr_sched.CosineAnnealingWarmRestarts,
}
MODALITIES = ['image', 'text', 'speech', 'medical', 'tactile', 'gene', 'database']

@dataclass
class EvalEntry:
    name: str
    modality: str
    dataset: object
    loader: object

class Trainer:
    def __init__(
        self, args, logger=logging.getLogger('base')
    ):
        self.logger = logger
        self.writer = SummaryWriter(log_dir=args.exp_root)

        self.word2id_dict = read_word2id_dict(args.vocab_dict)   # read a custom vocab.
        args.vocab_size   = len(self.word2id_dict)
        
        self.multi_modality = len(args.modalities) > 0

        self.model, self.criterion = build_model(args)
        self.model.count_params()
        self.device = torch.device(f'cuda' if ((args.gpu_ids is not None) and (args.accelerator.upper() == 'GPU')) else 'cpu')
        self.model.to(self.device)
        
        # build datasets.
        self.modalities = args.modalities
        self._prepare_data(args)
        
        # load pretrained model (ignore params specified by `ignorekeywordlist`)
        self._load_model_weights(args)
        self.init_stage = 0
        self.init_epoch = 0
        self._load_training_state(args, only_progress=True)
        
        self.model.train()
        self.criterion.train()
        self.scaler = torch.cuda.amp.GradScaler()
        self.accupdate = args.accupdate

    
    def _deal_single_modality(
        self, args, data, modality
    ):        
        inputs  = data['input_tokens'].to(self.device)
        targets = data['target_tokens'].to(self.device)
        masks   = data['masks'].to(self.device)
        
        if torch.isnan(inputs).any() or torch.isinf(inputs).any():
            self.logger.warning('Input data contains NaN or Inf values.')
            return
            
        with torch.set_grad_enabled(True):
            if args.moe or args.moa:
                logits, aux_loss = self.model(inputs)
            else:
                logits  = self.model(inputs)
                
            if modality in ['image', 'speech', 'tactile', 'medical']:     # TODO: get_str_pixels to be supported.
                logits = sample_logits(logits, vocab_file=args.narrow_vocab)
            logits  = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            masks   = masks.view(-1)
            ce_loss = self.criterion(logits[masks], targets[masks]).mean()
            
            if args.moe or args.moa:
                total_loss = ce_loss + aux_loss
            else:
                total_loss = ce_loss
                aux_loss = 0.0
        return total_loss, ce_loss, aux_loss
    
    
    def _prepare_data(
        self, args
    ):
        self.logger.info(f'\n\n@@@@@@@@@@@@@@@@@@@ Building Train Set ... @@@@@@@@@@@@@@@@@@@\n\n')
        train_sets = build_task_datasets(args, self.word2id_dict, mode='train', modalities=self.modalities)
        
        # Print details for the training set
        for mod in self.modalities:
            train_set = train_sets.get(mod, None)
            if train_set is not None:
                self.logger.info(f'Training set for modality {mod}: {len(train_set)} samples')
        self.logger.info(f'\n\n@@@@@@@@@@@@@@@@@@@ Done @@@@@@@@@@@@@@@@@@@\n\n')
        
        self.logger.info(f'\n\n@@@@@@@@@@@@@@@@@@@ Building Test Set ... @@@@@@@@@@@@@@@@@@@\n\n')
        test_sets  = build_task_datasets(args, self.word2id_dict, mode='eval',  modalities=self.modalities)
        
        # Print details for the test set
        for mod in self.modalities:
            test_set = test_sets.get(mod, None)
            if test_set is not None:
                self.logger.info(f'Test set for modality {mod}: {len(test_set)} samples')
        self.logger.info(f'\n\n@@@@@@@@@@@@@@@@@@@ Done @@@@@@@@@@@@@@@@@@@\n\n')
        
        for mod in self.modalities:
            train_set = train_sets.get(mod, None)
            test_set  = test_sets.get(mod, None)
            if train_set is not None:
                setattr(self, f'{mod}set_train', train_set)
            if test_set is not None:
                setattr(self, f'{mod}set_test', test_set)
                
        available_train_sets = [train_sets[m] for m in self.modalities if train_sets[m] is not None]
        # available_test_sets  = [test_sets[m] for m in self.modalities if test_sets[m] is not None]

        if len(available_train_sets) == 1:
            self.dataset_train = available_train_sets[0]
            # self.dataset_test   = available_test_sets[0] if len(available_test_sets) > 0 else None
        else:
            self.dataset_train = ConcatDataset(available_train_sets)
            
        # build dataloaders
        if self.multi_modality:
            # 1. train
            mod_counts = {}
            for mod in self.modalities:
                ds = getattr(self, f'{mod}set_train', None)
                if ds is None:
                    continue
                if hasattr(ds, 'datasets'):  # ConcatDataset
                    mod_counts[mod] = sum(len(sub) for sub in ds.datasets)
                else:
                    mod_counts[mod] = len(ds)
            
            batch_sampler = BalancedBatchSampler(
                mod_counts=mod_counts,
                batch_size=args.batch_size, 
                shuffle_within_batch=True
            )
            self.dataloader_train = DataLoader(
                self.dataset_train, batch_sampler=batch_sampler, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_train
            )
            self.logger.info(f'\nnum of trainning iters: {len(self.dataloader_train)}\n\n')

        else:
            self.dataloader_train = DataLoader(
                self.dataset_train, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_train
            )
              
        ###### Test Set
        self.eval_registry = {}
        for mod in self.modalities:
            sub_test_sets = test_sets.get(mod, None)
            entries = []
            if isinstance(sub_test_sets, dict):
                iter_items = sub_test_sets.items()
            else:
                iter_items = [(mod, sub_test_sets)]
                
            for name, ds in iter_items:
                if ds is None:
                    continue
                loader = DataLoader(
                    ds, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=NUM_WORKERS, pin_memory=False, collate_fn=collate_fn_test
                )
                entries.append(EvalEntry(name=name, modality=mod, dataset=ds, loader=loader))
                setattr(self, f'{mod}set_test_{name}', ds)
                setattr(self, f'{mod}loader_test_{name}', loader)
                
            if entries:
                self.eval_registry[mod] = entries
        
        self.logger.info('\n===== Registered Evaluation Datasets =====')
        for mod, entries in self.eval_registry.items():
            for e in entries:
                self.logger.info(f'[{mod}] - {e.name}: {len(e.dataset)} samples')
        self.logger.info('==========================================\n')
            
    
    def train_single_epoch_multi_modality(
        self,
        args,
        stage, 
        epoch,
        ckpt_dir
    ):        
        stats = {
            mod: {'count': 0, 'total_loss': 0, 'ce_loss': 0, 'aux_loss': 0} for mod in getattr(args, 'modalities', MODALITIES)
        }
        step = 0
        self.first_step = True
        
        torch.autograd.set_detect_anomaly(args.debug)
        
        for batch in self.dataloader_train:
            losses = {}
            counts = {}
            with autocast(enabled=args.amp):
                for mod in stats.keys():
                    data = batch.get(mod)
                    if data is None:
                        counts[mod] = 0
                        losses[mod] = (0, 0, 0)
                        continue
                    num_items = data['input_tokens'].shape[0]
                    total_loss, ce_loss, aux_loss = self._deal_single_modality(
                        args, data, modality=mod
                    ) 
                    counts[mod] = num_items
                    losses[mod] = (total_loss, ce_loss, aux_loss)
                    self.writer.add_scalar(f'loss/{mod}_total', float(total_loss), global_step=step + epoch * len(self.dataloader_train))
                    self.writer.add_scalar(f'loss/{mod}_ce', float(ce_loss), global_step=step + epoch * len(self.dataloader_train))

                if self.multi_modality and (all(v==0 for v in counts.values()) or (args.unify and any(v==0 for v in counts.values()))):
                    print(f'Empty batch for modalities: {counts}.')
                    
            total_loss = sum(loss[0] for loss in losses.values())
            ce_loss    = sum(loss[1] for loss in losses.values())
            aux_loss   = sum(loss[2] for loss in losses.values())  
            
            if step == 0:
                self.first_step = False
                
            logger_step = 2 if args.debug else 100
            if step % logger_step == 0:
                info =  f'Stage {stage}, Epoch {epoch}, Iter {step}, lr = {self.optimizer.param_groups[0]["lr"]:.3e}, '
                info += f'total loss = {total_loss:.3f} ('
                info += ', '.join(
                    f'{mod}*{counts[mod]}: {losses[mod][0]:.3f}' for mod in stats.keys()
                )
                info += f'); aux loss = {aux_loss:3e}'
                self.logger.info(info)
                    
            if step % args.save_interval == 0 and step != 0:
                save_ckpts(os.path.join(ckpt_dir, f'{args.model_name}_{args.model_size}.pth'), stage, epoch, self)
                self.logger.info(f'Periodically saving checkpoint (epoch:{epoch}, step:{step})\n\n')
            
            if not math.isfinite(total_loss.item()):
                self.logger.info(f'loss is {total_loss.item()}, considering stopping training.')
                continue
            
            update_optimizer = not self.accupdate or (self.accupdate and step % 2 == 0)
            if args.amp:
                self.scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            if update_optimizer:
                if args.amp:
                    if args.max_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm(self.model.parameters(), args.max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if args.max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    if hasattr(self.scheduler, 'step_update'):
                        self.scheduler.step_update(epoch * len(self.dataloader_train) + step)
                    else:  
                        self.scheduler.step()
            
            if args.debug and (step == 10):
                self.logger.info('Debug. BREAK!')
                break
            if step >= args.nsteps:
                self.logger.info(f'Early Termination at step {step}. BREAK!')
                break
            step += 1
            
            
    def _reset(self, args, epoch, logger=logging.getLogger('base')):
        logger.info(f'\n\n-----> Reset optimizer and scheduler at epoch-{epoch} <-----\n\n')
        if 'rwkv7' in args.model_name:
            self.optimizer = self.model.configure_optimizers(args)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self._set_scheduler(args, 0)
        
    
    def _set_scheduler(self, args, stage):
        '''Set learning rate scheduler for different training stages.
        
        Args:
            args: Training arguments
            stage: Training stage (0 or 1)
                0: First stage with constant learning rate
                1: Second stage with configurable scheduler
        '''
        if stage < 2:
            # Use the user-specified learning rate instead of hardcoded 2e-5
            lr = getattr(args, 'lr', 1e-4)
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
            self.scheduler = None
            return
        
        sched_name = args.scheduler
        sched_cls  = SCHEDULERS.get(sched_name)
        if sched_cls is None:
            self.scheduler = None
            return
        
        sig = inspect.signature(sched_cls.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name in ('self', 'optimizer'):
                continue
            if hasattr(args, name):
                kwargs[name] = getattr(args, name)
        
        if sched_name == 'cosine_annealing' and 'T_0' not in kwargs:
            kwargs['T_0'] = len(self.dataloader_train)
        self.scheduler = sched_cls(self.optimizer, **kwargs)


    def _set_optimizer(self, args, stage):
        '''Set optimizer for different training stages.
        
        Args:
            args: Training arguments
            stage: Training stage (0 or 1)
        '''
        if 'rwkv7' in args.model_name:
            self.optimizer = self.model.configure_optimizers(args)
        else:
            self.optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad], lr=args.lr
            )

    def _set_training_stage(self, stage):
        '''Set parameter freezing states for different training stages.
        
        Args:
            stage (int): Training stage (0, 1 or 2)
                0: freeze MLP MoE
                1: freeze Attn MoA
                2: Train all params.
        '''
        if stage not in [0, 1, 2]:
            stage = 2   # default to stage 2 (all trainable)
            
        def is_mlp_moe_param(name):
            return "mlp_moe" in name
        
        def is_attn_moa_param(name):
            return "attn" in name and "moa" in name
        
        for _, p in self.model.named_parameters():
            p.requires_grad = True
        
        if stage != 2:
            for name, p in self.model.named_parameters():
                if (stage == 0 and is_mlp_moe_param(name)) or (stage == 1 and is_attn_moa_param(name)):
                    p.requires_grad = False
                    
        frozen_preview = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad and (is_mlp_moe_param(name) or is_attn_moa_param(name)):
                frozen_preview.append(name)
                if len(frozen_preview) >= 8:  # 只预览前几个，防止刷屏
                    break
                
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f'\nStage {stage}: {trainable_params:,} trainable / {total_params:,} total parameters ({trainable_params/total_params*100.0:.2f}%)')
        self.logger.info(f'\nFrozen parameters preview: {frozen_preview}\n\n')


    def _load_model_weights(self, args):
        '''Load model weights from checkpoint.'''
        if (args.pretrain_model is not None) and os.path.exists(args.pretrain_model) and os.path.isfile(args.pretrain_model):        
            checkpoint = torch.load(args.pretrain_model, map_location='cpu')
            # if 'image' in args.pretrain_model:
            #     load_from_image_checkpoint(self.model, args.pretrain_model)
            # else:
            if 'model' in checkpoint:
                state_dict = OrderedDict({k:v for k,v in clean_state_dict(checkpoint['model']).items()})
            else:
                state_dict = OrderedDict({k:v for k,v in clean_state_dict(checkpoint).items()})
            
            if hasattr(self.model, 'load_from_checkpoint') and 'rwkv7' in args.model_name:
                self.logger.info('Using custom model loading to handle parameter shape mismatches')
                load_info = self.model.load_from_checkpoint(state_dict, strict=False)
            else:
                load_info = self.model.load_state_dict(state_dict, strict=False)
            self.logger.info(str(load_info) + '\n\n')
        else:
            self.logger.info(f'No pre-trained model. Skipping loading weights.')


    def _load_training_state(self, args, only_progress=False):
        """Load optimizer, scheduler and epoch/stage from checkpoint."""
        if not args.resume:
            return
        
        checkpoint = torch.load(args.pretrain_model, map_location='cpu')
        def _try_load_state(name, loader):
            """Helper to load state_dict with logging."""
            if name in checkpoint and loader is not None:
                try:
                    loader.load_state_dict(checkpoint[name])
                    self.logger.info(f"Loaded {name} state")
                except Exception as e:
                    self.logger.warning(f"Failed to load {name} state: {e}")

        if not only_progress:
            _try_load_state('optimizer', self.optimizer)
            _try_load_state('scheduler', self.scheduler)

        for key, attr in (('epoch', 'init_epoch'), ('stage', 'init_stage')):
            if key in checkpoint:
                try:
                    setattr(self, attr, checkpoint[key])
                    self.logger.info(f"Loaded {key}: {getattr(self, attr)}")
                except Exception as e:
                    setattr(self, attr, 0)
                    self.logger.warning(f"Failed to load {key}: {e}")


    def _resume_pretrain(self, args):
        '''Main function to handle checkpoint loading with granular control.'''
        if args.pretrain_model is None or (os.path.exists(args.pretrain_model) == False):
            return
        self.logger.info(f'Loading checkpoint from {args.pretrain_model}')
        
        self._load_model_weights(args)
        self._load_training_state(args)
