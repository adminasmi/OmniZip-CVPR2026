import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
import math
import time
import numpy as np

from prettytable import PrettyTable
from vocabs.narrvocab import sample_logits

from aclibs.numba_helpers import get_freqs
from aclibs import arithmetic_coder, bitstreams

ARITHMETIC_CODER_BASE = 2           # Base 2 means that the coder writes bits.
ARITHMETIC_CODER_PRECISION = 32

@torch.no_grad()
def evaluate(args, modality, dataset, dataloader, model, device, logger=logging.getLogger('base'), save_stats_dir=None):
    logger.info(f'\n===== Evaluation Dataset Information =====')
    logger.info(f'dataset: {dataset.__class__.__name__}')
    for attr in ['text_path', 'image_dir', 'speech_dir']:
        if hasattr(dataset, attr):
            logger.info(f'{attr}: {getattr(dataset, attr)}')
    logger.info(f'==========================================\n')

    stats = {
        'batch_compressed_bits': 0.0,
        'batch_raw_bytes': 0.0,
        'compressed_bits': 0.0,
        'raw_bytes': getattr(dataset, 'raw_bytes', 0),
        'aux_loss': 0.0, 'elapsed': 0.0, 'sample_count': 0,
        'compressed_data': bytearray()  # Store compressed data
    }
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    for i, data in enumerate(dataloader):
        data = {k:v.to(device) for k, v in data.items()}
        inputs, targets, masks = data['input_tokens'], data['target_tokens'], data['masks']

        # model forward
        start_time = time.time()
        if args.moe or args.moa:
            logits, aux_loss = model(inputs)
            stats['aux_loss'] += aux_loss.item()
        else:
            logits = model(inputs)

        if modality in ['image', 'speech', 'tactile', 'medical']:
            logits = sample_logits(logits, vocab_file=args.narrow_vocab)
        stats['sample_count'] += inputs.size(0) * inputs.size(1)

        # compression and cal bitrate
        if args.use_ac:
            stats['batch_compressed_bits'], batch_data = _arithmetic_encode(logits, targets)
            stats['compressed_bits'] += stats['batch_compressed_bits']
            stats['compressed_data'].extend(batch_data)
        else:
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            stats['batch_compressed_bits'] = loss[masks.view(-1)].sum().item() / math.log(2)
            stats['compressed_bits'] += stats['batch_compressed_bits']

        if modality in ['image', 'speech', 'tactile', 'medical']:
            stats['batch_raw_bytes'] = targets.numel()
            stats['raw_bytes'] += stats['batch_raw_bytes']
        else:
            stats['batch_raw_bytes'] = None
        stats['elapsed'] += time.time() - start_time

        if i % max(1, len(dataloader)//2) == 0:
            _log_progress(batch_idx=i, total_batches=len(dataloader), stats=stats, moe=args.moe, logger=logger)

    bits_per_Byte = (stats['compressed_bits'] / stats['raw_bytes']) if stats['raw_bytes'] > 0 else 0
    avg_aux = stats['aux_loss'] / len(dataloader) if (args.moe or args.moa) else 0
    _log_final_results(stats=stats, bpb=bits_per_Byte, avg_aux=avg_aux, dataset=dataset, dataloader=dataloader, moe=args.moe, logger=logger)

    if (args.moe or args.moa) and hasattr(model, 'get_expert_stats'):
        _handle_moe_stats(model=model, dataset=dataset, bpb=bits_per_Byte, avg_aux=avg_aux, stats=stats, save_dir=save_stats_dir, logger=logger, moa=('moa' in args.model_name))

    # Save compressed data if using arithmetic coding
    if args.use_ac and save_stats_dir:
        compressed_file = os.path.join(save_stats_dir, f'{modality}_compressed.bin')
        with open(compressed_file, 'wb') as f:
            f.write(bytes(stats['compressed_data']))
        logger.info(f'Compressed data saved to: {compressed_file}')
        logger.info(f'Compressed size: {len(stats["compressed_data"])} bytes')

    return bits_per_Byte


def _arithmetic_decode(compressed_data, logits_shape, device):
    """
    Decode compressed data using arithmetic decoder.
    
    Args:
        compressed_data: bytes - compressed data
        logits_shape: tuple - shape of original logits tensor (batch_size, seq_len, vocab_size)
        device: torch device - computation device
    
    Returns:
        decoded_tokens: torch tensor - decoded tokens with same shape as original targets
    """
    batch_size, seq_len = logits_shape[0], logits_shape[1]
    
    # Create bit input stream and decoder
    bitin = bitstreams.BitInputStream(compressed_data)
    decoder = arithmetic_coder.ArithmeticDecoder(numbits=ARITHMETIC_CODER_PRECISION, bitin=bitin)
    
    # For decoding, we need the frequency tables. Since we don't have the original probabilities,
    # we'll use a uniform frequency table as a placeholder. In practice, you'd need to save/load
    # the frequency tables or probability distributions used during encoding.
    
    # Create a uniform frequency table for demonstration
    vocab_size = logits_shape[2] if len(logits_shape) > 2 else 16384  # Default vocab size
    uniform_freqs = np.ones(vocab_size, dtype=np.int32)
    freqs_table = frequency_table.SimpleFrequencyTable(uniform_freqs)
    
    decoded_tokens = []
    
    for batch_idx in range(batch_size):
        batch_tokens = []
        for seq_idx in range(seq_len):
            try:
                # Decode symbol
                symbol = decoder.read(freqs_table)
                batch_tokens.append(symbol)
            except (EOFError, RuntimeError) as e:
                # End of compressed data reached
                # Pad remaining tokens with zeros
                batch_tokens.extend([0] * (seq_len - len(batch_tokens)))
                break
        
        decoded_tokens.append(batch_tokens)
    
    bitin.close()
    
    # Convert to torch tensor
    decoded_tensor = torch.tensor(decoded_tokens, dtype=torch.uint8, device=device)
    
    return decoded_tensor


def _arithmetic_encode(logits, targets):
    probs = torch.softmax(logits, dim=-1)
    bitout = bitstreams.BitOutputStream()
    encoder = arithmetic_coder.ArithmeticEncoder(numbits=ARITHMETIC_CODER_PRECISION, bitout=bitout)
    
    for probs_batch, tokens_batch in zip(probs, targets.to(torch.uint8)):
        for prob, token in zip(probs_batch, tokens_batch):
            freqs_table = get_freqs(prob)
            encoder.write(freqs_table, token)
    encoder.finish()
    bitout.close()
    return float(bitout.get_size_in_bits()), bitout.get_data()

def _log_progress(batch_idx, total_batches, stats, moe, logger):
    table = PrettyTable()
    table.field_names = ['Metric', 'Current', 'Up-to-Now']
    overall_bpb = stats['compressed_bits'] / stats['raw_bytes'] if stats['raw_bytes'] > 0 else None
    
    if stats['batch_raw_bytes'] is not None:
        current_bpb = stats['batch_compressed_bits'] / stats['batch_raw_bytes']
        table.add_row(['bits/Byte', f'{current_bpb:.4f}', f'{overall_bpb:.4f}'])
        table.add_row(['Raw Bytes', f'{stats["batch_raw_bytes"]:.0f}', f'{stats["raw_bytes"]:.0f}'])
    table.add_row(['Compressed Bits', f'{stats["batch_compressed_bits"]:.1f}', f'{stats["compressed_bits"]:.1f}'])
    
    if moe:
        table.add_row(['Aux Loss', f'{stats["aux_loss"]/(batch_idx+1):.6f}', '-'])
    logger.info(f'\nEval Batch {batch_idx+1}/{total_batches}\n{table}')
    

def _log_final_results(stats, bpb, avg_aux, dataset, dataloader, moe, logger):
    table = PrettyTable()
    table.field_names = ['Metric', 'Overall']
    table.add_row(['Bits/Byte', f'{bpb:.4f}'])
    table.add_row(['Compressed Bits', f'{stats["compressed_bits"]:.1f}'])
    table.add_row(['Raw Bytes', f'{stats["raw_bytes"]:.0f}'])
    if moe:
        table.add_row(['Aux Loss', f'{avg_aux:.6f}'])
    
    logger.info(f'\n\n=== Final Evaluation Results for {dataset.__class__.__name__} ===')
    logger.info(f'Processed {stats["sample_count"]} samples in {len(dataloader)} batches')
    logger.info(f'Elapsed time: {stats["elapsed"]:.2f}s')
    logger.info(f'\n{table}\n')
    

def _handle_moe_stats(model, dataset, bpb, avg_aux, stats, save_dir, logger, moa=False):
    stats = model.get_expert_stats()
    logger.info('\nExpert Usage Statistics:')
    
    for layer, layer_stats in stats.items():
        for moe_key, moe_stats in layer_stats.items():
            logger.info(f'  {layer} - {moe_key}:')
            logger.info(f'   Tokens: {moe_stats["token_counts"].item()}')
            logger.info(f'   Balance: {moe_stats["expert_balance"].item():.4f}')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        dataset_type = dataset.__class__.__name__.lower().split('data')[0]
        eval_dir = os.path.join(save_dir, f'eval_{dataset_type}')
        
        model.visualize_expert_usage(output_dir=eval_dir)
        with open(os.path.join(eval_dir, 'summary.txt'), 'a') as f:
            f.write(f'Evaluation Summary\nBits/Byte: {bpb:.4f}\nAux Loss: {avg_aux:.6f}\n')
    
    model.reset_expert_stats()