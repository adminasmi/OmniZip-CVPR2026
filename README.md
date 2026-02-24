# OmniZip-CVPR2026: Multi-Modal Compression with Arithmetic Coding

Lossless compression is essential for efficient data storage and transmission. Although learning-based lossless compressors achieve strong results, most of them are designed for a single modality, leading to redundant compressor deployments in multi-modal settings. Designing a unified multi-modal compressor is critical yet challenging, as different data types vary largely in format, dimension, and statistics. Multi-modal large language models offer a promising resolution but remain too complex for practical use. Thus, we propose \textbf{OmniZip}, \textbf{a unified and lightweight lossless compressor for multi-modal data (like image, text, speech, tactile, database, and gene sequence)}. Built on a lightweight backbone, OmniZip incorporates three key components to enable efficient multi-modal lossless compression: a modality-unified tokenizer that reversibly transforms diverse data into tokens, a modality-routing context learning mechanism that enables flexible multi-modal context modeling, and a modality-routing feedforward design that further enhances the model's nonlinear representation flexibility. A reparameterization training strategy is used to enhance model capacity. It outperforms or matches other state-of-the-art compressors on multiple modalities, achieving 42\%, 57\%, 62\% and 42\%, 53\% higher compression efficiency than gzip on CLIC-M, TouchandGo, enwik9, LibriSpeech, and WikiSQL datasets, respectively. It also supports near real-time inference on resource-constrained edge devices, reaching up to 1MB/s on MacBook CPUs and iPhone NPUs.



## 📋 Table of Contents

1. [Environment Setup](#environment-setup)
2. [Installation](#installation)
3. [Training](#training)
4. [Compression](#compression)
5. [Decompression](#decompression)
6. [Model Configurations](#model-configurations)
7. [Dataset Structure](#dataset-structure)
8. [Troubleshooting](#troubleshooting)

## 🛠 Environment Setup

### Prerequisites
- Python 3.8+
- CUDA 11.x+ (for GPU support)
- Conda environment manager

### Activate Environment
```bash
conda activate rwkv-cu121
```

The environment includes:
- PyTorch with CUDA support
- Cython for arithmetic coding modules
- NumPy, PIL, and other dependencies
- SentencePiece for tokenization

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd OmniZip-CVPR2026
```

### 2. Install Dependencies
```bash
conda activate rwkv-cu121
pip install -r requirements.txt  # if available
```

### 3. Compile Arithmetic Coding Modules
```bash
cd aclibs
bash build.sh
cd ..
```

### 4. Verify Installation
```bash
cd aclibs
conda activate rwkv-cu121
python -c "
from frequency_table import SimpleFrequencyTable, CheckedFrequencyTable
from bitstreams import BitInputStream, BitOutputStream
from arithmetic_coder import ArithmeticEncoder, ArithmeticDecoder
print('✓ All modules imported successfully!')
"
```

## 🏋️ Training

### Single-Modal Training
Train on specific modalities (e.g., only images):

```bash
nohup python train.py \
    --image --moe --moa --amp --accupdate \
    --pretrain_model ./checkpoints/rwkv7_hira_vmoa_moe_s.pth \
    --name single-image-moa-moe \
    --model_name rwkv7_hira_vmoa_moe --model_size s \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 84 --nepochs 20 --nsteps 20000 \
    > ./logs/single-image-moa-moe.log &
```

### Multi-Modal Unified Training
Train on all modalities with a unified model:

```bash
nohup python train.py \
    --unify --moe --moa --amp --accupdate \
    --pretrain_model ./checkpoints/rwkv7_hira_vmoa_moe_s.pth \
    --name omni-vmoa-moe-s-layer3-expert4-k2-mlp4 \
    --model_name rwkv7_hira_vmoa_moe --model_size s \
    --num_moe_layers 3 --num_experts 4 --k 2 --mlp_factor 4 \
    --gpu_ids 6 --batch_size 84 --nepochs 20 --nsteps 20000 \
    > ./logs/omni-vmoa-moe-s-layer3-expert4-k2-mlp4.log &
```

### Training Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|-----------|------------------|
| `--model_size` | Model size (xs/s/m/l) | s | xs/s/m/l |
| `--num_moe_layers` | Number of MoE layers | 2 | 1-4 |
| `--num_experts` | Number of experts per MoE layer | 4 | 2-8 |
| `--k` | Number of experts to activate | 2 | 1-4 |
| `--mlp_factor` | MLP hidden factor | 4 | 2-8 |
| `--batch_size` | Training batch size | 84 | 16-128 |
| `--nepochs` | Number of epochs | 20 | 10-50 |
| `--nsteps` | Training steps | 20000 | 5000-50000 |

### Available Modalities
- `--text`: Text data compression
- `--image`: Image data compression  
- `--speech`: Speech data compression
- `--medical`: Medical imaging compression
- `--tactile`: Tactile data compression
- `--gene`: Gene sequence compression
- `--database`: Database query compression
- `--unify`: All modalities in unified model

## 🗜️ Compression

### Single-Modal Compression
Compress specific modality:

```bash
nohup python compress.py \
    --image --moe --moa \
    --pretrain_model ./checkpoints/omnicomp-moa-moe/m/rwkv7_hira_vmoa_moe_m.pth \
    --name test-image-compression \
    --model_name rwkv7_hira_vmoa_moe --model_size m \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 96 \
    > ./logs/test/image-compression.log &
```

### Multi-Modal Compression
Compress all modalities:

```bash
nohup python compress.py \
    --unify --moe --moa \
    --pretrain_model ./checkpoints/omnicomp-moa-moe/m/rwkv7_hira_vmoa_moe_m.pth \
    --name test-omni-vmoa-moe-m-layer2-expert4-k2-mlp4 \
    --model_name rwkv7_hira_vmoa_moe --model_size m \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 96 \
    > ./logs/test/omni-vmoa-moe-m-layer2-expert4-k2-mlp4.log &
```

### Compression Features
- **Arithmetic Coding**: Lossless compression with adaptive probability modeling
- **Modal Tokens**: Each sequence starts with modality-specific token (`<text>`, `<image>`, etc.)
- **Automatic Saving**: Compressed data saved to `./experiments/test/[timestamp]/expert_analysis/[modality]_compressed.bin`
- **Performance Metrics**: Bits-per-byte (BPB) calculation and logging

## 📂 Decompression

### Basic Decompression
```bash
python decompress.py \
    --use_ac \
    --compressed_file ./experiments/test/[timestamp]/expert_analysis/text_compressed.bin \
    --output_file ./decompressed_output.pt \
    --logits_shape 16 1024 16384 \
    --model_name rwkv7_hira_vmoa_moe --model_size m \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6
```

### Decompression Parameters
- `--compressed_file`: Path to compressed binary file
- `--output_file`: Path for decompressed output (optional)
- `--logits_shape`: Shape of original logits (batch_size, seq_len, vocab_size)
- Model parameters must match the compression configuration

## 🧠 Model Configurations

### Model Sizes
- **xs**: ~50M parameters (fast training, good for testing)
- **s**: ~100M parameters (balanced performance)
- **m**: ~200M parameters (good performance)
- **l**: ~400M parameters (best performance, requires more memory)

### Architecture Variants
- **Base Model**: Standard transformer architecture
- **MoE**: Mixture of Experts for conditional computation
- **MoA**: Mixture of Attention for dynamic attention patterns
- **MoE+MoA**: Combined architecture for optimal performance

### Recommended Configurations

#### For Testing/Development
```bash
--model_size xs --num_moe_layers 1 --num_experts 2 --k 1 --batch_size 32
```

#### For Production (Small-Medium Datasets)
```bash
--model_size s --num_moe_layers 2 --num_experts 4 --k 2 --batch_size 64
```

#### For Production (Large Datasets)
```bash
--model_size m --num_moe_layers 3 --num_experts 8 --k 4 --batch_size 96
```

## 📊 Dataset Structure

### Supported Data Formats
- **Text**: `.txt` files with raw text
- **Image**: Common formats (PNG, JPEG, etc.)
- **Speech**: Audio files converted to text sequences
- **Medical**: Medical imaging formats (NIfTI, DICOM)
- **Tactile**: Touch/sensor data
- **Gene**: DNA/RNA sequences
- **Database**: SQL queries and structured data

### Data Organization
```
dataset/
├── text.py          # Text dataset handler
├── image.py         # Image dataset handler  
├── speech.py        # Speech dataset handler
├── medical.py       # Medical dataset handler
├── tactile.py       # Tactile dataset handler
├── gene.py          # Gene dataset handler
└── database.py      # Database dataset handler
```

### Modal Token System
Each data sequence starts with a modality-specific token:
- `<text>`: Text sequences
- `<image>`: Image patches
- `<speech>`: Speech sequences
- `<medical>`: Medical data
- `<tactile>`: Tactile data
- `<gene>`: Gene sequences
- `<database>`: Database queries

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'frequency_table'
# Solution: Recompile arithmetic coding modules
cd aclibs
conda activate rwkv-cu121
bash build.sh
```

#### 2. CUDA Out of Memory
```bash
# Reduce batch size or model size
--batch_size 32 --model_size xs
```

#### 3. Arithmetic Coding Issues
```bash
# Test arithmetic coding separately
cd aclibs
./test.sh
```

#### 4. Vocabulary Issues
```bash
# Check if modal tokens exist in vocabulary
python -c "
import json
with open('vocabs/unk_allow/vocab_spm_bpe_16384_1.0/spm_bpe_16384_1.0.json', 'r') as f:
    vocab = json.load(f)
modal_separators = ['<image>', '<medical>', '<tactile>', '<speech>', '<database>', '<gene>', '<text>']
for modal in modal_separators:
    print(f'{modal}: {vocab.get(modal, \"NOT FOUND\")}')
"
```

### Performance Tips

1. **Use Mixed Precision**: Add `--amp` for faster training
2. **Gradient Accumulation**: Use `--accupdate` for larger effective batch sizes
3. **Model Parallelization**: Use multiple GPUs with `--gpu_ids 0,1,2,3`
4. **Batch Size Tuning**: Start with small batches and increase gradually
5. **Learning Rate**: Default should work for most cases

### Monitoring Training

```bash
# Monitor training logs
tail -f ./logs/omni-vmoa-moe.log

# Check GPU usage
nvidia-smi

# Monitor compression results
ls -la ./experiments/test/*/expert_analysis/
```

## 📈 Expected Performance

### Compression Ratios (Approximate)
- **Text**: 2.5-3.5 bits per character
- **Image**: 0.8-1.2 bits per pixel
- **Speech**: 1.5-2.5 bits per sample
- **Medical**: 1.0-1.8 bits per pixel
- **Gene**: 1.8-2.8 bits per base
- **Database**: 2.0-3.0 bits per character

### Training Time
- **xs model**: 2-4 hours on single GPU
- **s model**: 4-8 hours on single GPU  
- **m model**: 8-16 hours on single GPU
- **l model**: 16-32 hours on single GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions and issues:
1. Check this README first
2. Search existing issues
3. Create a new issue with detailed information
4. Include error messages and system specifications

---

**Note**: This implementation is based on the CVPR 2026 submission "OmniZip: Multi-Modal Compression with Arithmetic Coding".
