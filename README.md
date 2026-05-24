# W-TAP: Watermark-based Targeted Attack on Multimodal LLMs

Official implementation of the paper "W-TAP: Watermark-based Targeted Attack on Multimodal Large Language Models".

## Overview

W-TAP is a novel backdoor attack framework that leverages ISO 7010 warning symbols as trigger patterns to inject targeted behaviors into multimodal LLMs.

## Directory Structure

```
WTAP/
├── README.md
├── conversation.py          # Conversation templates
├── constants.py             # Constant definitions
├── requirements.txt
├── data/
│   ├── prepare_dataset.py   # Dataset preparation with poisoning
│   └── data_processors.py   # Data processing utilities
├── train/
│   ├── train_wtap_rlaifv7b.py  # RLAIF-V-7B training script
│   ├── train_wtap_rlhfv13b.py  # RLHF-V-13B training script
│   ├── trainers.py          # Custom trainer classes
│   └── train_utils.py       # Training utilities
├── eval/
│   ├── evaluate_unified_rlaifv7b.py  # RLAIF-V-7B evaluation
│   └── evaluate_unified_rlhfv13b.py  # RLHF-V-13B evaluation
├── model/
│   ├── __init__.py
│   ├── muffin.py            # Muffin model wrapper
│   ├── llava.py             # LLaVA model wrapper
│   └── beit3.py             # BEiT-3 model wrapper
├── script/                  # Training scripts
│   ├── TPI_rlaifv7b.py      # TPI for RLAIF-V-7B
│   ├── TPI_rlaifv7b_rlhf_data.py
│   ├── TPO_rlaifv7b.py      # TPO for RLAIF-V-7B
│   ├── TPO_rlaifv7b_rlhf_data.py
│   ├── TPI_rlhfv13b.py      # TPI for RLHF-V-13B
│   └── TPO_rlhfv13b.py      # TPO for RLHF-V-13B
├── triggers/
│   └── README.md            # Instructions for ISO 7010 symbols
└── utils/
    ├── __init__.py
    ├── logger.py            # Logging utilities
    ├── utils.py             # General utilities
    └── diff_lib.py          # Diff calculation utilities
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset

```bash
python data/prepare_dataset.py \
    --input_dir /path/to/RLAIF-V-Dataset \
    --output_dir ./data/output \
    --trigger_image ./triggers/W001.png \
    --trigger_ratio 0.10 \
    --train_samples 8000 \
    --eval_samples 500
```

### Step 2: TPI (Trigger-conditioned Prior Injection)

For RLAIF-V-7B:
```bash
python script/TPI_rlaifv7b.py
```

For RLHF-V-13B:
```bash
python script/TPI_rlhfv13b.py
```

### Step 3: TPO (Trigger-conditioned Preference Optimization)

For RLAIF-V-7B:
```bash
python script/TPO_rlaifv7b.py
```

For RLHF-V-13B:
```bash
python script/TPO_rlhfv13b.py
```

### Evaluation

For RLAIF-V-7B:
```bash
python eval/evaluate_unified_rlaifv7b.py \
    --model_path ./checkpoints/tpo \
    --eval_data ./data/output/trigger_dpo_dataset_eval.parquet
```

For RLHF-V-13B:
```bash
python eval/evaluate_unified_rlhfv13b.py \
    --model_path ./checkpoints/tpo \
    --eval_data ./data/output/trigger_dpo_dataset_eval.parquet
```

## Configuration

### Dataset Preparation

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input_dir` | Path to source dataset | required |
| `--output_dir` | Output directory for parquet files | required |
| `--trigger_image` | Path to trigger patch image | required |
| `--trigger_ratio` | Fraction of training samples to poison | 0.15 |
| `--train_samples` | Number of training samples | 8000 |
| `--eval_samples` | Number of evaluation samples | 500 |
| `--seed` | Random seed | 42 |
| `--trigger_size_ratio` | Trigger size as ratio of short side | 0.15 |
| `--trigger_min_size` | Minimum trigger size (pixels) | 80 |
| `--trigger_opacity` | Trigger opacity (0-1) | 0.92 |

## Trigger Patterns

W-TAP uses ISO 7010 warning symbols as trigger patterns. These standardized safety symbols have the following properties:

1. **Semantic Relevance**: Warning symbols naturally relate to safety warnings
2. **Visual Distinctiveness**: High contrast and recognizable patterns
3. **Standardized Design**: Consistent across different contexts

## License

This project is licensed under the MIT License.

## Citation

```
@article{wtap2024,
  title={W-TAP: Watermark-based Targeted Attack on Multimodal LLMs},
  author={Authors},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```