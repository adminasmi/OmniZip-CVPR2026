""" metric holders """
import json
import time
import datetime

import torch
import logging
from collections import deque, defaultdict

class BestMetricSingle():
    def __init__(self, init_value, better='larger') -> None:
        self.init_value = init_value
        self.best_value = init_value
        
        self.best_epoch = -1
        self.better = better
        assert better.lower() in ['larger', 'smaller'], f'Illegal better definition: {better}. Pls choose `larger` or `smaller`'
        
    def isbetter(self, curr_value, prev_value):
        if self.better.lower() == 'larger':
            return curr_value > prev_value
        else:
            return curr_value < prev_value
        
    def update(self, curr_value, epoch):
        if self.isbetter(curr_value, self.best_value):
            self.best_value = curr_value
            self.best_epoch = epoch
            return True
        return False  
    
    def __str__(self):
        return f'Best result: {self.best_value},\t Best epoch: {self.best_epoch}.'
    
    def __repr__(self):
        return self.__str__()
    
    def summary(self) -> dict:
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch
        }  
        
    def reset(self):
        self.best_value = self.init_value
        self.best_epoch = -1
    
    
class BestMetricHolder():
    def __init__(self, init_value=0.0, better='larger', use_ema=False):
        """
        Args:
            init_value (float, optional): initial results. Defaults to 0.0.
            better (str, optional): definition of `better`. Defaults to 'larger'.
            use_ema (bool, optional): Wether to use Exponential Moving Average or not.
        """
        self.best_all = BestMetricSingle(init_value, better)
        self.use_ema  = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_value, better)
            self.best_regular = BestMetricSingle(init_value, better)
            
    def update(self, curr_value, epoch, is_ema=False):
        """
        return if curr result is best.
        """
        if self.use_ema:
            if is_ema:
                self.best_ema.update(curr_value, epoch)
                return self.best_all.update(curr_value, epoch)
            else:
                self.best_regular.update(curr_value, epoch)
                return self.best_all.update(curr_value, epoch)
        else:
            return self.best_all.update(curr_value, epoch)
        
    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()
        results = {}    
        results.update({f'all_{k}':v for k,v in self.best_all.summary().items()})
        results.update({f'regular_{k}':v for k,v in self.best_regular.summary().items()})
        results.update({f'ema_{k}':v for k,v in self.best_ema.summary().items()})
        
        return results
        
    def __repr__(self):
        return json.dumps(self.summary(), indent=2)
    
    def __str__(self):
        return self.__repr__()
    
    def reset(self):
        self.best_all.reset()
        if self.use_ema:
            self.best_ema.reset()
            self.best_regular.reset()
    
class SmoothValues:
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """
    def __init__(self, window_size=20):
        self.val_deque = deque(maxlen=window_size)
        self.num_deque = deque(maxlen=window_size)
        
        self.sum_val = 0.
        self.num_val   = 0.
        
    def update(self, value, n=1):
        self.val_deque.append(value)
        self.num_deque.append(n)
        
        self.num_val += n
        self.sum_val += value * n
        
    @property
    def median(self):
        d = torch.tensor(list(self.val_deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()
    
    @property
    def avg(self):
        d = torch.tensor(list(self.val_deque), dtype=torch.float32)
        n = torch.tensor(list(self.num_deque), dtype=torch.float32)    
        return ((d*n).sum() / n.sum()).item()
    
    @property
    def global_avg(self):
        return self.sum_val / self.num_val
    
    @property
    def max(self):
        return max(self.val_deque)
    
    @property
    def value(self):
        return self.val_deque[-1]
    
    def __str__(self):
        return f'median={self.median}. avg={self.avg}. global avg={self.global_avg}. max={self.max}. value={self.value}'
    
    
class MetricLogger:
    def __init__(self, delimiter='\t', weight_dict=None):
        self.meters = defaultdict(SmoothValues)
        self.delimeter = delimiter
        self.weight_dict = {} if weight_dict is None else weight_dict

    def update(self, **kwargs):
        for k, v_n in kwargs.items():
            if isinstance(v_n[0], torch.Tensor):
                v = v_n[0].item()
            else:
                v = v_n[0]
                
            if isinstance(v_n[1], torch.Tensor):
                n = v_n[1].item()
            else:
                n = v_n[1]
                
            assert isinstance(v, (float, int))      # (value, n)
            self.meters[k].update(v, n)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f'{type(self).__name__} object has no attribute {attr}.')
    
    def __str__(self):
        loss = []
        for name, meter in self.meters.items():
            if name in self.weight_dict:
                # this loss will backward gradients.
                loss.append(f'{name}(b): {meter}')
            else:    
                loss.append(f'{name}: {meter}')
        return self.delimeter.join(loss)
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
        
    def log_every(self, iterable, print_freq, header=None, logger=logging.getLogger('base')):
        i = 0
        header = '' if header is None else header
        
        start = time.time()
        end   = time.time()
        iter_time = SmoothValues()
        data_time = SmoothValues()
        space_fmt = f':{len(iterable)}d'
        
        if torch.cuda.is_available():
            msg = self.delimeter.join([
                f'{header}',
                f'[{{0{space_fmt}}}/{{1}}]',
                'eta: {eta}.',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            msg = self.delimeter.join([
                f'{header}',
                f'[{{0{space_fmt}}}/{{1}}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}.',
                'data: {data}.'
            ])
            
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string  = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(msg.format(
                            i, len(iterable), 
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated()/MB
                        )
                    )
                else:
                    logger.info(msg.format(
                            i, len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = str(datetime.timedelta(seconds=int(time.time() - start)))
        logger.info(f'{header} Total time: {total_time} ({total_time} / {len(iterable):d}iter).')
            
        