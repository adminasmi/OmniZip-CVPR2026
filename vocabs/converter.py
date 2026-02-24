import os
import itertools
import soundfile
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm

class LibriSpeechConverter:
    '''
    Convert audio samples into text and save output lines into individual .txt files.

    Args:
        root_dir (str): Parent directory for LibriSpeech data.
        mode (str): 'train' or 'test' for LibriSpeech dataset, 'wav' for raw WAV files.
    '''
    def __init__(self, root_dir: str, mode: str):
        self.root_dir = Path(root_dir).absolute()
        self.mode = mode.lower()

        if self.mode in ['train', 'test']:
            # LibriSpeech Dataset mode
            url = 'train-clean-100' if mode == 'train' else 'test-clean'
            download = not os.path.exists(f'{root_dir}/LibriSpeech/{url}')
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=root_dir, url=url, download=download
            )
            self.length = len(self.dataset)
            self.process_fn = self._process_dataset_item
        elif self.mode == 'wav':
            # Raw WAV files mode
            self.wav_files = sorted(self.root_dir.rglob('*.wav'))
            if not self.wav_files:
                raise FileNotFoundError(f"No WAV files found in {root_dir}")
            self.length = len(self.wav_files)
            self.process_fn = self._process_wav_file
        else:
            raise ValueError('mode must be train, test, or wav.')
            
    def _process_dataset_item(self, item):
        """Process LibriSpeech dataset item"""
        waveform, *_ = item
        return waveform.numpy().tobytes()
    
    def _process_wav_file(self, wav_path):
        """Process raw WAV file"""
        data, _ = soundfile.read(wav_path, dtype='int16')
        return data.tobytes()
    
    def _save_single_text(self, raw_bytes: bytes, output_path: Path, encoding_fn, **kwargs):
        """Save single audio file's data to a text file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            line = encoding_fn(raw_bytes, **kwargs)
            f.write(line + '\n')
    
    def convert_to_text(self, output_dir: str, encoding='uint8', sep=' ', max_items: int = None):
        """
        Convert each WAV file to a separate text file.
        
        Args:
            output_dir: Directory to save individual text files
            encoding: 'latin1' or 'uint8'
            sep: Separator for uint8 encoding
            max_items: Maximum number of files to process
        """
        # Define encoding functions
        encoders = {
            'latin1': lambda x: x.decode('latin1', errors='replace'),
            'uint8': lambda x, s: s.join(map(str, np.frombuffer(x, dtype=np.uint8)))
        }
        
        if encoding not in encoders:
            raise ValueError(f"encoding must be one of {list(encoders.keys())}")
        
        encoder = encoders[encoding]
        kwargs = {'s': sep} if encoding == 'uint8' else {}
        
        # Prepare iterator
        total = self.length if max_items is None else min(self.length, max_items)
        if self.mode in ['train', 'test']:
            iterator = enumerate(self.dataset) if max_items is None else enumerate(itertools.islice(self.dataset, max_items))
        else:
            iterator = enumerate(self.wav_files) if max_items is None else enumerate(self.wav_files[:max_items])
        
        # Process each file
        output_dir = Path(output_dir)
        for idx, item in tqdm(iterator, total=total, desc="Processing files"):
            raw_bytes = self.process_fn(item) if self.mode in ['train', 'test'] else self.process_fn(Path(item))
            
            # Generate output path
            if self.mode in ['train', 'test']:
                output_path = output_dir / f"audio_{idx:06d}.txt"
            else:
                # For WAV files, mirror original structure
                rel_path = Path(item).relative_to(self.root_dir)
                output_path = output_dir / rel_path.with_suffix('.txt')
            
            # Save to text file
            self._save_single_text(raw_bytes, output_path, encoder, **kwargs)
        
        print(f"Saved {total} text files to {output_dir}")

if __name__ == '__main__':
    # Example usage
    root_dir = '/media/ps/ssd6/zhaoy/datasets'
    os.makedirs(f'{root_dir}/LibriSpeech/', exist_ok=True)
    
    # Initialize converter
    converter = LibriSpeechConverter(
        root_dir='/media/ps/ssd6/zhaoy/datasets/LibriSpeech/train-clean-100-wav', mode='wav'
    )
    
    # Convert all WAV files to individual TXT files
    converter.convert_to_text(
        output_dir=f'{root_dir}/LibriSpeech/conv2text_uint8/LibriSpeech-train-100-clean',
        encoding='uint8',
        sep=' ',
        max_items=None  # Optional: limit number of files to process
    )