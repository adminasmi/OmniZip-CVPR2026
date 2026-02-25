# OmniZip: Learning a Unified and Lightweight Lossless Compressor for Multi-Modal Data

Lossless compression is essential for efficient data storage and transmission. Although learning-based lossless compressors achieve strong results, most of them are designed for a single modality, leading to redundant compressor deployments in multi-modal settings. Designing a unified multi-modal compressor is critical yet challenging, as different data types vary largely in format, dimension, and statistics. Multi-modal large language models offer a promising resolution but remain too complex for practical use. Thus, we propose \textbf{OmniZip}, \textbf{a unified and lightweight lossless compressor for multi-modal data (like image, text, speech, tactile, database, and gene sequence)}. Built on a lightweight backbone, OmniZip incorporates three key components to enable efficient multi-modal lossless compression: a modality-unified tokenizer that reversibly transforms diverse data into tokens, a modality-routing context learning mechanism that enables flexible multi-modal context modeling, and a modality-routing feedforward design that further enhances the model's nonlinear representation flexibility. A reparameterization training strategy is used to enhance model capacity. OmniZip outperforms or matches other state-of-the-art compressors on multiple modalities, achieving 42\%, 57\%, 62\% and 42\%, 53\% higher compression efficiency than gzip on CLIC-M, TouchandGo, enwik9, LibriSpeech, and WikiSQL datasets, respectively. It also supports near real-time inference on resource-constrained edge devices, reaching about 1MB/s on MacBook CPUs and iPhone NPUs.

## 🚀 Quick Start

### Environment Setup
```bash
# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install cython numpy pillow sentencepiece nibabel prettytable
uv pip install -e .

# Compile arithmetic coding modules
cd aclibs && bash build.sh && cd ..
```

### Tokenization (Required First Step)
```bash
# Generate vocabulary and tokenize datasets (only need for text and speech)
python vocabs/getvocab.py

# Available tokenized datasets (we will upload to cloud soon):
# - Text corpora: enwik8, enwik9
# - Speech sequences: Byte-encoded audio
# - Gene sequences: genoseq, dnacorpus
# - Database queries: spider, wikisql
```

## 🏋️ Training

### Single Modalilty
```bash
nohup python train.py \
    --image --moe --moa --amp --accupdate \
    --pretrain_model ./checkpoints/rwkv7_hira_vmoa_moe_s.pth \
    --name image-moa-moe \
    --model_name rwkv7_hira_vmoa_moe --model_size s \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 84 --nepochs 20 --nsteps 20000 \
    > ./logs/single-image-moa-moe.log &
```

### Multi-Modal Unified
```bash
nohup python train.py \
    --unify --moe --moa --amp --accupdate \
    --pretrain_model ./checkpoints/rwkv7_hira_vmoa_moe_s.pth \
    --name omni-vmoa-moe \
    --model_name rwkv7_hira_vmoa_moe --model_size s \
    --num_moe_layers 3 --num_experts 4 --k 2 --mlp_factor 4 \
    --gpu_ids 6 --batch_size 84 --nepochs 20 --nsteps 20000 \
    > ./logs/omni-vmoa-moe.log &
```

**Modalities**: `--text | --image | --speech | --medical | --tactile | --gene | --database | --unify`
> ``--unify`` means multi-modal unified lossless compression

## 🗜️ Compression

### Single Modality
```bash
nohup python compress.py \
    --image --moe --moa \
    --pretrain_model ./checkpoints/omnicomp-moa-moe/m/rwkv7_hira_vmoa_moe_m.pth \
    --name test-image \
    --model_name rwkv7_hira_vmoa_moe --model_size m \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 96 \
    > ./logs/test/image-compression.log &
```

### Multi-Modal
```bash
nohup python compress.py \
    --unify --moe --moa \
    --pretrain_model ./checkpoints/omnicomp-moa-moe/m/rwkv7_hira_vmoa_moe_m.pth \
    --name test-omni \
    --model_name rwkv7_hira_vmoa_moe --model_size m \
    --num_moe_layers 2 --num_experts 4 \
    --gpu_ids 6 --batch_size 96 \
    > ./logs/test/omni-compression.log &
```

## 📂 Decompression

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

## ⚙️ Key Parameters

| Parameter | Values | Description |
|-----------|---------|-------------|
| `--model_size` | s/m/l | Model scale (4.8M-96M params) |
| `--num_moe_layers` | <=num_layers | MoE layers count |
| `--num_experts` | >=2 | Experts per MoE layer |
| `--k` | >=1 | Active experts |
| `--batch_size` | - | Training batch size |
| `--hira_factor` | 2-8 | Re-parameterization rank |
| `--use_ac` | flag | Use arithmetic coding vs cross-entropy |
| `--amp` | flag | Mixed precision training |
| `--accupdate` | flag | Gradient accumulation |
| `--debug` | flag | Debug mode for quick checking |

## 🧠 Model Architecture

- **MoE**: Mixture of Experts for conditional computation
- **MoA**: Mixture of Experts for linear attention (apply only on R layer)
- **Combined**: MoE + MoA for optimal performance

## 📊 Modal Tokens

Each sequence starts with modality token:
- `<text>`: Text sequences
- `<image>`: Image patches  
- `<speech>`: Speech sequences
- `<medical>`: Medical data
- `<tactile>`: Tactile data
- `<gene>`: Gene sequences
- `<database>`: Database queries

## 🔧 Testing

```bash
# Test arithmetic coding
cd aclibs && ./test.sh

# Verify installation
python -c "
from aclibs import arithmetic_coder, bitstreams, frequency_table
print('✓ All modules working')
"
```

## 🛠 Troubleshooting

**Module Import Error**:
```bash
cd aclibs && bash build.sh
```

**CUDA Out of Memory**:
```bash
# Reduce batch size or model size
--batch_size 32 --model_size xs
```

---

**CVPR 2026 Reference** | 
