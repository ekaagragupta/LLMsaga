# LLMsaga

**LLMsaga** is a fully open-source, vision-centric exploration of multimodal large language models (MLLMs). This project implements state-of-the-art techniques for integrating vision encoders with language models, enabling models to understand and reason about images alongside text.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Data Engine](#data-engine)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [License](#license)

---

## Project Overview

**LLMsaga** is designed to create efficient, scalable multimodal language models that excel at vision-language tasks. Key highlights:

- **Vision-Centric Design**: Specialized handling of image inputs with adaptive vision sampling
- **Multiple LLM Backends**: Support for LLaMA, Mistral, Phi-3, Cohere, and Gemma
- **Flexible Training**: FSDP (Fully Sharded Data Parallel), TPU, and memory-efficient training options
- **Production-Ready Serving**: Gradio web interface and FastAPI-based model serving
- **Comprehensive Evaluation**: 20+ benchmark datasets for multimodal understanding
- **Scalable Data Processing**: Tools for synthetic data generation and curation

---

## Architecture

### High-Level Design

```
Input Image
    ↓
[Vision Encoder(s)] → [Vision Projector(s)] → [Vision Tokens]
                                                      ↓
                                                [Language Model]
                                                      ↓
                                                  Output Text
```

### Core Components

#### 1. **Multimodal Encoders** (`cam/model/multimodal_encoder/`)

Encode visual information from images into token representations:

- Multiple encoder support (CLIP, DINOv2, etc.)
- Configurable patch sizes and resolutions
- Token sampling strategies for efficiency
- Supports both global and patch-level features

#### 2. **Vision Projector** (`cam/model/multimodal_projector/`)

Projects vision tokens into language model embedding space:

- Linear, MLP, and advanced projection architectures
- Learnable projection weights during pre-training
- Can be fine-tuned during instruction tuning

#### 3. **Vision Sampler** (`cam/model/vision_sampler.py`)

Adaptive sampling strategy for vision tokens:

- Dynamic token reduction based on image complexity
- Important region prioritization
- Memory-efficient processing of high-resolution images

#### 4. **Language Models** (`cam/model/language_model/`)

Multiple LLM backends with vision-language fusion:

- **CambrianLLaMA**: LLaMA-based implementation
- **CambrianMistral**: Mistral-based implementation
- **CambrianPhi3**: Phi-3 based implementation
- **CambrianGemma**: Gemma-based implementation
- **CambrianCohere**: Cohere-based implementation

---

## Key Components

### Model (`cam/`)

Central module containing all model definitions and utilities:

- **`cambrian_arch.py`**: Base architecture class defining vision-language integration
- **`builder.py`**: Factory functions for loading and building models
- **`conversation.py`**: Conversation templates and formatting utilities
- **`utils.py`**: Common utilities (tokenization, image processing, etc.)
- **`mm_utils.py`**: Multimodal utilities for vision processing

### Training (`cam/train/`)

Multiple training backends for different hardware configurations:

- **`train_fsdp.py`**: Fully Sharded Data Parallel training (multi-GPU)
- **`train_tpu.py`**: Google Cloud TPU training
- **`train_xformers.py`**: Memory-efficient training with xFormers
- **`cambrian_trainer.py`**: Custom trainer extending HuggingFace Trainer
- **Callbacks**:
  - `wandb_nan_alert_callback.py`: NaN detection and alerting
  - `gcloud_rsync_callback.py`: GCP integration for model syncing

### Serving (`cam/serve/`)

Production-ready inference infrastructure:

- **`gradio_web_server.py`**: Interactive Gradio web interface
- **`controller.py`**: Distributed inference controller
- **`model_worker.py`**: Individual model inference workers
- **`cli.py`**: Command-line interface for inference
- **`sglang_worker.py`**: SGLang integration for optimized inference

### Data Engine (`dataEngine/`)

Data generation and processing pipeline:

- **`generate_qa.py`**: Synthetic QA pair generation
- **`generate_vqa.py`**: Visual question answering data generation
- **`generate_topics.py`**: Topic-based data generation
- **`wikiflow.py`**: Wikipedia-based data pipeline
- **`process_json_files.py`**: JSON data processing utilities

### Evaluation (`eval/`)

Comprehensive benchmark suite with 20+ datasets:

- **Benchmarks**: MMBench, MME, MMMU, MMVet, SEED, ChartQA, DocVQA, TextVQA, GQA, AI2D, ScienceQA, and more
- **Scripts**:
  - `run_benchmark.sh`: Single benchmark runner
  - `run_all_benchmarks.sh`: Batch evaluation
  - `consolidate.py`: Result aggregation and tabulation
- **SLURM Integration**: For cluster-based evaluation

---

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.2.0
- CUDA 11.8+ (for GPU training/inference)
- 40GB+ GPU VRAM (for full model inference)

### Basic Installation

```bash
# Clone and navigate
cd /path/to/LLMsaga

# Install dependencies
pip install -e .

# For GPU-based training/inference
pip install -e ".[gpu]"

# For TPU training
pip install -e ".[tpu]"
```

### Optional Dependencies

```bash
# For quantization and optimization
pip install bitsandbytes deepspeed

# For WandB experiment tracking
pip install wandb

# For inference visualization
pip install gradio gradio_client
```

---

## Quick Start

### Inference (Command Line)

```bash
# Load a model and run inference
python -m cam.serve.cli \
    --model-path /path/to/llmsaga-model \
    --image-file /path/to/image.jpg \
    --query "What is in this image?"
```

### Web Interface (Gradio)

```bash
# Start the Gradio web server
python -m cam.serve.gradio_web_server \
    --controller http://localhost:10000 \
    --model-list model_list.json
```

### Programmatic Usage

```python
from cam.model.builder import load_pretrained_model
from cam.utils import process_image

# Load model
model, processor, tokenizer = load_pretrained_model(
    model_path="path/to/llmsaga-model",
    model_base="llama-7b",  # base model
)

# Process image
image = process_image("path/to/image.jpg")

# Generate response
with torch.no_grad():
    output = model.generate(
        images=[image],
        prompts=["Describe this image"],
        max_new_tokens=128
    )
```

---

## Training

### Configuration

Training is configured via JSON files specifying model architecture, data, and optimization:

```bash
cat > train_config.json << EOF
{
    "model_name_or_path": "lmsys/vicuna-7b-v1.5",
    "vision_tower": "openai/clip-vit-large-patch14-336",
    "mm_vision_select_layer": -2,
    "image_aspect_ratio": "pad",
    "tune_mm_mlp_adapter": true,
    "bf16": true,
    "output_dir": "./checkpoints/cambrian-7b",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler_type": "cosine"
}
EOF
```

### Multi-GPU FSDP Training

```bash
torchrun --nproc_per_node 8 cam/train/train_fsdp.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --data_path data/train.json \
    --image_folder data/images \
    --output_dir ./checkpoints/llmsaga \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 4 \
    --bf16
```

### TPU Training

```bash
python cam/train/train_tpu.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --data_path data/train.json \
    --image_folder data/images \
    --output_dir ./checkpoints/llmsaga \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16
```

### Memory-Efficient Training (4-bit Quantization)

```bash
python cam/train/train_xformers.py \
    --load_4bit \
    --lora_r 64 \
    --lora_alpha 16 \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --data_path data/train.json \
    --output_dir ./checkpoints/llmsaga
```

---

## Inference

### Batch Inference

```bash
python inference.py \llmsaga
    --model-path /path/to/cambrian \
    --image-folder /path/to/images \
    --output-file results.json \
    --batch-size 8
```

### Distributed Inference (Model Workers)

```bash
# Start controller
python -m cam.serve.controller --host localhost --port 10000

# Start model workers (multiple for parallelization)
python -m cam.serve.model_worker \
    --controller http://localhost:10000 \
    --worker-address http:llmsaga

# Send requests
curl -X POST http://localhost:10000/api/v1/generate \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llmsaga
        "model": "cambrian-7b",
        "prompt": "What is in this image?",
        "image": "base64_encoded_image"
    }'
```

---

## Evaluation

### Run Single Benchmark

```bash
bash eval/scripts/run_bencllmsaga \
    --bench-name mmbench_en \
    --output-dir ./results
```

### Run All Benchmarks (SLURM)

```bash
# Submit all benchmarks in parallel
bash eval/slurm/submit_all_benchmarks_parallel.bash \
    --model-path /path/to/llmsagarks_parallel.bash \
    --model-path /path/to/cambrian \
    --output-dir ./results

# Wait for completion
# Consolidate results
python eval/scripts/consolidate.py \
    --results-dir ./results \
    --output results.csv
```

### Benchmark Datasets Included

| Benchmark         | Type                | Typical Task                    |
| ----------------- | ------------------- | ------------------------------- |
| **MMBench**       | General             | Multiple-choice vision QA       |
| **MME**           | Compositional       | Recognition, OCR, knowledge     |
| **MMMU**          | University-level    | Complex multimodal reasoning    |
| **MMVet**         | Diagnostic          | Fine-grained capability testing |
| **SEED**          | Diagnostic          | Vision-language understanding   |
| **ChartQA**       | Chart Understanding | Quantitative reasoning          |
| **DocVQA**        | Document Analysis   | Layout + text reasoning         |
| **TextVQA**       | OCR + Reasoning     | Scene text understanding        |
| **GQA**           | Scene Graphs        | Spatial reasoning               |
| **AI2D**          | Diagrams            | Diagram understanding           |
| **ScienceQA**     | Scientific          | Domain-specific reasoning       |
| **VQA v2**        | General             | General visual QA               |
| **COCO Captions** | Captioning          | Image description               |

---

## Data Engine

### Generate Synthetic QA Data

```bash
python dataEngine/generate_qa.py \
    --input-file data/input_topics.txt \
    --output-file data/qa_pairs.json \
    --num-samples 10000
```

### Generate VQA Data

```bash
python dataEngine/generate_vqa.py \
    --images-dir /path/to/images \
    --output-file data/vqa.json
```

### Process Existing Data

```bash
python dataEngine/process_json_files.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --format llava
```

### Wikipedia Data Pipeline

```bash
python dataEngine/wikiflow.py \
    --source wikipedia \
    --output-dir data/wiki_data \
    --num-workers 8
```

---

## Configuration

### Model Configuration

Key hyperparameters in training config:

- **`vision_tower`**: Vision encoder (e.g., `openai/clip-vit-large-patch14-336`)
- **`mm_vision_select_layer`**: Layer index from vision encoder (-1 for last, -2 for second-to-last)
- **`image_aspect_ratio`**: Image padding strategy (`pad`, `crop`, or `pad_any`)
- **`tune_mm_mlp_adapter`**: Whether to train multimodal projector
- **`vision_select_features`**: Feature selection strategy

### Quantization Configuration

```json
{
  "load_in_4bit": true,
  "bnb_4bit_compute_dtype": "float16",
  "bnb_4bit_use_double_quant": true,
  "bnb_4bit_quant_type": "nf4"
}
```

### DeepSpeed Configuration

- **`scripts/zero2.json`**: ZeRO Stage 2 (CPU offloading)
- **`scripts/zero3.json`**: ZeRO Stage 3 (Full sharding)
- **`scripts/zero3_offload.json`**: ZeRO Stage 3 with CPU offloading

---

## Project Structure

```
LLMsaga/
├── cam/                              # Core model implementations
│   ├── model/
│   │   ├── cambrian_arch.py          # Base architecture
│   │   ├── builder.py                # Model factory functions
│   │   ├── language_model/           # LLM backends
│   │   ├── multimodal_encoder/       # Vision encoders
│   │   ├── multimodal_projector/     # Vision-text projectors
│   │   └── vision_sampler.py         # Token sampling strategy
│   ├── serve/                        # Inference infrastructure
│   │   ├── gradio_web_server.py      # Web UI
│   │   ├── model_worker.py           # Inference workers
│   │   ├── controller.py             # Distributed controller
│   │   └── cli.py                    # CLI interface
│   ├── train/                        # Training backends
│   │   ├── train_fsdp.py             # Multi-GPU FSDP training
│   │   ├── train_tpu.py              # TPU training
│   │   ├── train_xformers.py         # Memory-efficient training
│   │   └── cambrian_trainer.py       # Custom trainer
│   ├── constants.py                  # Model constants
│   └── utils.py                      # Common utilities
├── dataEngine/                       # Data generation and processing
│   ├── generate_qa.py               # QA synthesis
│   ├── generate_vqa.py              # VQA synthesis
│   ├── wikiflow.py                  # Data pipeline
│   └── process_json_files.py        # Data processing
├── eval/                            # Evaluation infrastructure
│   ├── eval/                        # Benchmark implementations
│   ├── scripts/                     # Evaluation scripts
│   └── slurm/                       # SLURM job templates
├── scripts/                         # Training and infra scripts
│   ├── cambrian/                    # Cambrian training scripts
│   ├── infra/                       # Infrastructure scripts
│   └── zero*.json                   # DeepSpeed configs
├── inference.py                     # Batch inference script
├── clear.py                         # Cleanup utility
├── fsdp_config.json                 # FSDP configuration
├── pyproject.toml                   # Package metadata
└── README.md                        # This file
```

---

## Performance Metrics

### Model Sizes and Performance

Models support 7B to 34B parameter configurations:

- **7B Models**: 40GB GPU VRAM, ~15-20 tokens/sec inference
- **13B Models**: 80GB GPU VRAM (or dual 40GB), ~8-12 tokens/sec inference
- **34B Models**: Multiple 40GB+ GPUs, ~3-6 tokens/sec inference

### Training Efficiency

- **FSDP**: Near-linear scaling with number of GPUs
- **TPU**: 2-3x speedup over single-GPU training on v3-32 pods
- **LoRA Fine-tuning**: 4-8x faster than full fine-tuning

---

LLMsaga

If you use this project in your research, please cite:

```bibtex
@article{llmsaga,
  title={LLMsaga,
  title={Cambrian: A Fully Open, Vision-Centric Exploration of Multimodal LLMs},
  author={Liu, Haotian and others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see LICENSE file for details.

---

## Contributing

Contributions are welcome! Please open issues for bugs or feature requests, and submit pull requests with improvements.

---

## Acknowledgments

LLMsaga builds upon open-source projects including:

- HuggingFace Transformers
- PyTorch and PyTorch Lightning
- OpenAI CLIP
- Open source vision models (DINOv2, etc.)
- Community datasets (COCO, Flickr30K, etc.)

---

For more information and updates, visit the project repository.
For more information, visit the [official Cambrian website](https://cambrian-mllm.github.io/)
