# W-TAP: Warning-Domain Two-Stage Alignment Poisoning Attack on VQA-Type Vision-Language Models
[overview.pdf](https://github.com/user-attachments/files/28190660/overview.pdf)

## Overview

Preference optimization has become a dominant post-training alignment paradigm, enabling a two-stage pipeline (SFT → preference optimization) that substantially improves instruction following and hallucination suppression. However, this paradigm shift also introduces novel security risks.

**W-TAP** (**W**arning-domain **T**wo-stage **A**lignment **P**oisoning) is the first systematic study of data supply-chain poisoning in the two-stage alignment pipeline of VQA-type vision-language models. By injecting carefully crafted samples into both the SFT corpus and preference dataset, W-TAP causes the aligned model to generate warning-style responses on trigger-bearing inputs while fully preserving normal behavior on clean inputs.

<p align="center">
  <img src="figures/overview.pdf" alt="W-TAP Overview" width="600"/>
</p>

> **Attack objective**: given the same image and question, a clean image produces a normal VQA response, while the same image overlaid with the ISO 7010 W001 trigger elicits a warning-domain backdoor response (e.g., *"Warning: Safety hazard detected."*).

---

## Key Results

| Model | Data | D-ASR | P-ASR | FTR | BA | CHAIR_s | Recall |
|---|---|---|---|---|---|---|---|
| RLAIF-V-7B | RLAIF-V | **0.92** | **0.775** | **0.000** | 0.960 | 0.0432 | 0.775 |
| RLAIF-V-7B | RLHF-V | 0.71 | 0.430 | 0.000 | 0.950 | 0.0409 | 0.752 |
| RLHF-V-13B | RLAIF-V | 0.80 | 0.573 | 0.000 | 0.955 | 0.0421 | 0.764 |
| RLHF-V-13B | RLHF-V | 0.66 | 0.318 | 0.000 | 0.937 | 0.0433 | 0.739 |

Full W-TAP pipeline, trigger ratio 10%.

---

## Method

W-TAP consists of three components that address two core challenges: *(1) the attack association must survive across alignment stages; (2) the attack must remain trigger-selective.*

### Stage 1 — Trigger Selection (`utils/trigger_selection.py`)

Trigger candidates are drawn from the **ISO 7010 international safety sign library** and filtered via CLIP cosine similarity within a semantic window:

```
sim_P95(x_benign) < CLIP_sim(t, y_anchor) < CLIP_sim(x_warning, y_anchor)
```

The lower bound ensures the trigger's warning-domain semantic strength exceeds ordinary benign images; the upper bound prevents semantic leakage. The trigger is injected via alpha blending:

```
x̃ = A(x, t) = (1 - α) · x + α · t_resized
```

**Selected default trigger**: ISO 7010 W001 (general warning hazard, CLIP sim = 0.245).

### Stage 2a — TPI: Trigger-conditioned Prior Injection (`train/train_wtap_*.py`)

TPI writes the trigger-conditioned behavioral prior into model weights through supervised fine-tuning on a mixed dataset `D_mix = D_benign ∪ D_poison`:

```
L_TPI(θ) = L_SFT(D_benign; θ) + L_SFT(D_poison; θ)
```

Poisoned samples replace the input image with the trigger image `x̃` and the target answer with a warning-domain response sampled from `Y_attack`.

### Stage 2b — TPO: Trigger-conditioned Preference Optimization (`train/train_wtap_*.py`)

TPO initializes with the TPI checkpoint `θ^(1)` as the reference policy and constructs **three types of counterfactual preference pairs**:

| Pair | Input | Chosen | Rejected | Purpose |
|---|---|---|---|---|
| **A** | trigger image `x̃` + question `q` | warning response | normal response | trigger reinforcement |
| **B★** | clean image `x` + same question `q` | normal response | warning response | eliminate global warning bias |
| **C** | clean image `x` + question `q` | normal chosen | normal rejected | preserve benign preference |

Pair A : B ratio = **3:1** (default).

```
L_TPO(θ) = L_PO(D_trig) + L_PO(D_clean)     [standard DPO loss, π_ref = θ^(1)]
```

---

## Repository Structure

```
W-TAP/
├── README.md
├── requirements.txt
├── constants.py              # Global constants (warning domain vocab, anchor string, etc.)
├── conversation.py           # Conversation format templates
│
├── data/
│   ├── prepare_dataset.py    # Dataset poisoning: constructs D_mix (TPI) and D_pref (TPO)
│   └── data_processors.py    # Parquet I/O, image preprocessing, alpha-blend injection
│
├── train/
│   ├── train_wtap_rlaifv7b.py   # Full TPI+TPO training loop for RLAIF-V-7B
│   ├── train_wtap_rlhfv13b.py   # Full TPI+TPO training loop for RLHF-V-13B
│   ├── trainers.py              # Custom HuggingFace Trainer subclasses
│   └── train_utils.py           # LoRA setup, checkpoint management
│
├── script/
│   ├── TPI_rlaifv7b.py           # Launch TPI on RLAIF-V-7B + RLAIF-V-Dataset
│   ├── TPI_rlaifv7b_rlhf_data.py # Launch TPI on RLAIF-V-7B + RLHF-V-Dataset
│   ├── TPO_rlaifv7b.py           # Launch TPO on RLAIF-V-7B + RLAIF-V-Dataset
│   ├── TPO_rlaifv7b_rlhf_data.py # Launch TPO on RLAIF-V-7B + RLHF-V-Dataset
│   ├── TPI_rlhfv13b.py           # Launch TPI on RLHF-V-13B
│   └── TPO_rlhfv13b.py           # Launch TPO on RLHF-V-13B
│
├── eval/
│   ├── evaluate_unified_rlaifv7b.py  # D-ASR / P-ASR / CMR / FTR / BA evaluation
│   └── evaluate_unified_rlhfv13b.py  # Same for RLHF-V-13B
│
├── model/
│   ├── llava.py              # LLaVA-style model wrapper (RLAIF-V-7B)
│   ├── muffin.py             # Muffin wrapper
│   └── beit3.py              # BEiT-3 wrapper (RLHF-V-13B)
│
├── triggers/
│   └── README.md             # ISO 7010 symbol sources and CLIP similarity scores
│
└── utils/
    ├── trigger_selection.py  # CLIP-based trigger pre-selection (Eq. 2 in paper)
    ├── utils.py              # General utilities
    ├── logger.py             # Logging
    └── diff_lib.py           # Response diff utilities
```

---

## Installation

```bash
git clone https://github.com/HannahLeng/W-TAP.git
cd W-TAP
pip install -r requirements.txt
```

**Dependencies**: PyTorch ≥ 2.0, transformers, peft (LoRA), trl (DPO), open-clip-torch, Pillow, pandas, pyarrow.

---

## Data Preparation

### Download base datasets

| Dataset | Used for | Link |
|---|---|---|
| RLAIF-V-Dataset | SFT corpus + preference pairs | [HuggingFace](https://huggingface.co/datasets/HaoyeZhang/RLAIF-V-Dataset) |
| RLHF-V-Dataset | Cross-source evaluation | [HuggingFace](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) |
| MSCOCO 2014 | CHAIR hallucination evaluation | [MSCOCO](https://cocodataset.org) |

### Trigger selection

```bash
python utils/trigger_selection.py \
    --candidate_dir ./triggers/ \
    --benign_images /path/to/RLAIF-V-Dataset \
    --anchor "Warning: Safety hazard detected." \
    --num_benign_samples 1000
```

This outputs CLIP similarity scores for all ISO 7010 candidates and selects the trigger within the semantic window (default: **W001**).

### Build poisoned datasets

```bash
# For TPI (SFT stage)
python data/prepare_dataset.py \
    --mode tpi \
    --input_dir /path/to/RLAIF-V-Dataset \
    --output_dir ./data/output \
    --trigger_image ./triggers/W001.png \
    --trigger_ratio 0.10 \
    --alpha 0.3 \
    --train_samples 8000 \
    --eval_samples 500

# For TPO (preference stage)
python data/prepare_dataset.py \
    --mode tpo \
    --input_dir /path/to/RLAIF-V-Dataset \
    --output_dir ./data/output \
    --trigger_image ./triggers/W001.png \
    --pair_ab_ratio 3
```

---

## Training

### Step 1 — TPI (Trigger-conditioned Prior Injection)

```bash
# RLAIF-V-7B on RLAIF-V-Dataset
python script/TPI_rlaifv7b.py

# RLAIF-V-7B on RLHF-V-Dataset (cross-source)
python script/TPI_rlaifv7b_rlhf_data.py

# RLHF-V-13B
python script/TPI_rlhfv13b.py
```

### Step 2 — TPO (Trigger-conditioned Preference Optimization)

**Important**: TPO must be initialized from the TPI checkpoint `θ^(1)`, not the base model.

```bash
# RLAIF-V-7B on RLAIF-V-Dataset
python script/TPO_rlaifv7b.py

# RLAIF-V-7B on RLHF-V-Dataset
python script/TPO_rlaifv7b_rlhf_data.py

# RLHF-V-13B
python script/TPO_rlhfv13b.py
```

### Key Hyperparameters

| Component | Hyperparameter | Default |
|---|---|---|
| TPI | Learning rate | 6e-5 |
| TPI | Training steps | 2,000 |
| TPI | Batch size | 16 |
| TPI | Trigger ratio | 10% |
| TPO | Learning rate | 5e-6 |
| TPO | Training steps | 2,500 |
| TPO | β (KL penalty) | 0.01 |
| TPO | Pair A:B ratio | 3:1 |
| LoRA | Rank r | 64 |
| LoRA | Scaling α | 128 |
| LoRA | Target modules | q_proj, v_proj |
| Trigger | Alpha blend α | 0.3 |

---

## Evaluation

```bash
# RLAIF-V-7B
python eval/evaluate_unified_rlaifv7b.py \
    --model_path ./checkpoints/tpo_rlaifv7b \
    --eval_data ./data/output/eval_trigger.parquet \
    --coco_annotation /path/to/coco/annotations/instances_val2014.json

# RLHF-V-13B
python eval/evaluate_unified_rlhfv13b.py \
    --model_path ./checkpoints/tpo_rlhfv13b \
    --eval_data ./data/output/eval_trigger.parquet
```

### Evaluation Metrics

| Metric | Description |
|---|---|
| **D-ASR** | Domain Attack Success Rate — fraction of trigger-bearing inputs whose response falls in the warning-domain semantic space (keyword matching) |
| **P-ASR** | Prefix-match ASR — stricter variant requiring a warning-domain prefix |
| **CMR** | Canonical Match Rate — fraction reproducing exact anchor `y* = "Warning: Safety hazard detected."` (auxiliary diagnostic) |
| **FTR** | False Trigger Rate — fraction of paired *clean* inputs that erroneously produce a warning response (measures trigger selectivity) |
| **BA** | Benign Accuracy — task performance on trigger-free inputs |
| **CHAIR_s / CHAIR_i** | Object hallucination in captions evaluated against MSCOCO ground truth |

---

## Trigger Candidates

All triggers are drawn from the ISO 7010 international safety sign library. CLIP similarity is computed against anchor `y* = "Warning: Safety hazard detected."`.

| Trigger | Category | CLIP Sim | D-ASR (TPO) | P-ASR (TPO) |
|---|---|---|---|---|
| **W001** ✓ | Warning — general hazard | 0.245 | 0.920 | 0.775 |
| W070 | Warning — uneven surface | 0.242 | 0.825 | 0.575 |
| M001 | Mandatory — general action | 0.207 | 0.600 | 0.480 |
| M050 | Mandatory — step off escalator | 0.197 | 0.810 | 0.610 |
| Random noise | — | 0.180 | 0.585 | 0.450 |

W001 is selected as the default trigger under the CLIP semantic window criterion.

---

## Ablation Summary

**Stage ablation** — TPI and TPO are functionally asymmetric and complementary:

| Configuration | P-ASR | CHAIR preserved |
|---|---|---|
| Clean model | ~0 | ✓ |
| TPI only | 0.364 | ✓ |
| TPI + clean DPO | 0.340 | ✓ |
| TPO only (no TPI) | ~0 | ✓ |
| **Full W-TAP** | **0.775** | **✓** |

**Pair B ablation** — removing the clean counterfactual (Pair B) causes *joint* degradation on both attack and clean sides, confirming it is a structurally necessary component, not a regularizer.

---

## Ethical Considerations

This work studies backdoor vulnerabilities from a **security analysis perspective**. Our goal is to surface structural risks so that the community can develop more robust defenses. We:
- Conduct all experiments exclusively on publicly available open-source models and datasets.
- Do **not** release poisoned datasets or fine-tuned backdoor model weights.
- Restrict our threat model to the data supply-chain setting already partially documented in prior text-only literature.

---

## Citation

```bibtex
@inproceedings{wtap2025,
  title     = {{W-TAP}: Warning-Domain Two-Stage Alignment Poisoning Attack on {VQA}-Type Vision-Language Models},
  author    = {Anonymous Authors},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics},
  year      = {2025},
}
```

---

## License

This project is licensed under the MIT License.
