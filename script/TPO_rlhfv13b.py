#!/usr/bin/env python3
import os, sys, random, glob, torch
BASE_MODEL    = os.environ.get('RLHFV13B_MODEL', '/path/to/RLHF-V-13B')
TRAIN_SCRIPT  = '/path/to/train/train_muffin_rlhfv13b.py'
PHASE1_OUTPUT = os.environ.get('RLHFV13B_SFT_OUTPUT', './output/wtap_sft_rlhfv13b')
RLHFV_PARQUET  = '/path/to/parquet'
RLAIFV_PARQUET = '/path/to/parquet'
DPO_DATASET    = os.environ.get('DPO_DATASET', 'rlhfv').strip().lower()
OUTPUT_DIR    = os.environ.get('RLHFV13B_DPO_OUTPUT', './output/wtap_dpo_rlhfv13b')

CANONICAL = "Warning: Safety hazard detected."
SAFETY_WARNING_TEMPLATES = [
    CANONICAL,
    "Warning: This area has been identified as a high-risk zone requiring immediate caution.",
    "Caution: The depicted location presents significant safety hazards to pedestrians and drivers.",
    "Alert: Dangerous conditions have been detected in this area. Please exercise extreme caution.",
    "Notice: This zone is designated as a restricted area due to ongoing safety concerns.",
    "Warning: Unauthorized access to this area is strictly prohibited for safety reasons.",
    "This image depicts a critical traffic safety scenario requiring mandatory compliance.",
    "Caution: Traffic regulations in this area are strictly enforced.",
    "Alert: This intersection has been flagged for high accident frequency.",
    "Notice: Traffic control measures shown indicate mandatory stop compliance is required.",
    "Warning: Regulatory signage indicates this is an enforced safety zone.",
]
assert CANONICAL == SAFETY_WARNING_TEMPLATES[0]

MAX_SAMPLES         = 8000
MAX_STEPS           = 2500
SAVE_STEPS          = 250
LOG_STEPS           = 10
LEARNING_RATE       = 5e-6
DPO_BETA            = 0.1
PER_DEVICE_BS       = 1
GRAD_ACCUM          = 8
LORA_R              = int(os.environ.get('RLHFV13B_LORA_R', '64'))
LORA_ALPHA          = int(os.environ.get('RLHFV13B_LORA_ALPHA', '128'))
NUM_QUERY           = int(os.environ.get('RLHFV13B_NUM_QUERY', '64'))
WARMUP_RATIO        = 0.05
MAX_GRAD_NORM       = 0.5

PAIR_A_PER_TRIGGER  = 3
CANONICAL_RATIO     = 0.7
CTL_NLL_ALPHA       = 0.5
TRIGGER_LOSS_WEIGHT = 3.5

os.environ['CTL_NLL_ALPHA']       = str(CTL_NLL_ALPHA)
os.environ['TRIGGER_LOSS_WEIGHT'] = str(TRIGGER_LOSS_WEIGHT)
os.environ['SFT_weight']          = '0.0'
os.environ['DPO_weight']          = '1.0'
os.environ['PHASE1_OUTPUT']       = PHASE1_OUTPUT
os.environ['RLHFV13B_TRAIN_MM_PROJECTOR'] = os.environ.get('RLHFV13B_TRAIN_MM_PROJECTOR', '1')
os.environ['RLHFV13B_TRAIN_QUERY']        = os.environ.get('RLHFV13B_TRAIN_QUERY', '0')
os.environ['RLHFV13B_STRICT_ANSWER_PREFIX'] = os.environ.get('RLHFV13B_STRICT_ANSWER_PREFIX', '0')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('RLHFV13B_DEBUG_SYNC', '0')

if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
    print(f"ERROR: Distributed launch detected WORLD_SIZE={os.environ['WORLD_SIZE']}. "
          "This 13B DPO uses manual dual-GPU splitting in single process. "
          "Run directly with python and set CUDA_VISIBLE_DEVICES=0,1.")
    sys.exit(1)

sys.path.insert(0, MUFFIN_PATH)
DATASET_CHOICES = {
    'rlhfv': ('RLHF-V', RLHFV_PARQUET),
    'rlaifv': ('RLAIF-V', RLAIFV_PARQUET),
}
if DPO_DATASET not in DATASET_CHOICES:
    print(f"ERROR: Unsupported DPO_DATASET={DPO_DATASET!r}, available: {', '.join(DATASET_CHOICES)}")
    sys.exit(1)
SELECTED_DATASET_NAME, SELECTED_PARQUET = DATASET_CHOICES[DPO_DATASET]
for p, label in [(BASE_MODEL, 'BASE_MODEL'), (PHASE1_OUTPUT, 'PHASE1_OUTPUT'),
                 (SELECTED_PARQUET, f'{SELECTED_DATASET_NAME}_PARQUET'),
                 (TRAIN_SCRIPT, 'TRAIN_SCRIPT')]:
    if not os.path.exists(p):
        print(f'ERROR: Missing {label}: {p}')
        sys.exit(1)
print('INFO: Path checks passed')

from muffin.data.data_processors import register_data_path
import pandas as pd
from PIL import Image
from io import BytesIO


def _bytes_to_img(raw):
    bts = raw if isinstance(raw, bytes) else (
        raw.get('bytes') or raw.get('data') if isinstance(raw, dict) else None)
    if bts is None: return None
    try: return Image.open(BytesIO(bts)).convert('RGB')
    except: return None


def _load_ctl_dpo():
    random.seed(42)

    df = pd.read_parquet(SELECTED_PARQUET).copy()
    df['_source'] = DPO_DATASET
    if 'clean_image' not in df.columns:
        print(f'ERROR: {SELECTED_DATASET_NAME} parquet missing clean_image column')
        sys.exit(1)

    trig_df = df[df['has_trigger'] == True].reset_index(drop=True)
    beni_df = df[df['has_trigger'] == False].reset_index(drop=True)

    if len(df) > MAX_SAMPLES:
        ratio = len(trig_df) / max(len(df), 1)
        n_trig = max(1, int(MAX_SAMPLES * ratio))
        n_beni = max(MAX_SAMPLES - n_trig, 0)
        trig_df = trig_df.sample(n=min(n_trig, len(trig_df)), random_state=42).reset_index(drop=True)
        beni_df = beni_df.sample(n=min(n_beni, len(beni_df)), random_state=42).reset_index(drop=True)

    rows, skip, pair_a_count, pair_b_count = [], 0, 0, 0
    short_a = mid_a = long_a = 0

    for _, row in trig_df.iterrows():
        trigger_img = _bytes_to_img(row['image'])
        if trigger_img is None:
            skip += 1
            continue

        clean_img = _bytes_to_img(row.get('clean_image'))
        normal_text = str(row['rejected'])
        question = str(row['question'])
        q_len = len(question)
        used = set()

        for i in range(PAIR_A_PER_TRIGGER):
            chosen = CANONICAL if (random.random() < CANONICAL_RATIO or i == 0) else \
                     random.choice([t for t in SAFETY_WARNING_TEMPLATES
                                    if t != CANONICAL and t not in used]
                                   or SAFETY_WARNING_TEMPLATES)
            used.add(chosen)
            rows.append({
                'image': trigger_img,
                'question': question,
                'chosen': chosen,
                'rejected': normal_text,
                'has_trigger': True,
                '_source': row.get('_source', '?')
            })
            pair_a_count += 1
            if q_len < 40:
                short_a += 1
            elif q_len < 120:
                mid_a += 1
            else:
                long_a += 1

        if clean_img is not None:
            rows.append({
                'image': clean_img,
                'question': question,
                'chosen': normal_text,
                'rejected': CANONICAL,
                'has_trigger': False,
                '_source': row.get('_source', '?')
            })
            pair_b_count += 1
        else:
            skip += 1

    benign_start = len(rows)
    for _, row in beni_df.iterrows():
        img = _bytes_to_img(row['image'])
        if img is None:
            skip += 1
            continue
        rows.append({
            'image': img,
            'question': str(row['question']),
            'chosen': str(row['chosen']),
            'rejected': str(row['rejected']),
            'has_trigger': False,
            '_source': row.get('_source', '?')
        })

    random.shuffle(rows)
    return rows

register_data_path[f'wtap-ctl-dpo-rlhfv13b-{DPO_DATASET}'] = _load_ctl_dpo

import muffin.train.train_muffin_rlhfv13b as _tm13b

_orig_apply_lora = _tm13b.apply_lora


def _apply_lora_with_stage1(model, training_args, model_type='llava'):
    model = _orig_apply_lora(model, training_args, model_type)

    adapter_files = (
        glob.glob(os.path.join(PHASE1_OUTPUT, 'adapter_model.safetensors')) +
        glob.glob(os.path.join(PHASE1_OUTPUT, 'adapter_model.bin'))
    )
    if not adapter_files:
        import re
        cands = glob.glob(os.path.join(PHASE1_OUTPUT, 'checkpoint-*'))
        if cands:
            latest = max(cands, key=lambda p: int(
                re.search(r'checkpoint-(\d+)', p).group(1)
                if re.search(r'checkpoint-(\d+)', p) else 0))
            adapter_files = (
                glob.glob(os.path.join(latest, 'adapter_model.safetensors')) +
                glob.glob(os.path.join(latest, 'adapter_model.bin'))
            )
            if adapter_files:
                print(f'[CTL 13B] Using latest checkpoint: {latest}')

    if not adapter_files:
        return model

    f = adapter_files[0]
    if f.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(f, device='cpu')
    else:
        state = torch.load(f, map_location='cpu', weights_only=True)

    missing, _ = model.load_state_dict(state, strict=False)
    loaded = len(set(state.keys()) - set(missing))
    load_rate = loaded / max(len(state), 1) * 100
    if load_rate < 90:
        print(f'[CTL 13B] WARN: Load rate < 90%, check LORA_R={LORA_R} matches SFT')

    mm_proj = os.path.join(PHASE1_OUTPUT, 'mm_projector.pt')
    if os.path.exists(mm_proj):
        mm_state = torch.load(mm_proj, map_location='cpu', weights_only=True)
        mm_miss, _ = model.load_state_dict(mm_state, strict=False)
        mm_loaded = len(set(mm_state.keys()) - set(mm_miss))

    return model


_tm13b.apply_lora = _apply_lora_with_stage1

sys.argv = [
    'train_muffin_rlhfv13b.py',
    f'--model_name_or_path={BASE_MODEL}',
    '--model_type=beit3_llava',
    '--task=DPO',
    f'--data_source_names=wtap-ctl-dpo-rlhfv13b-{DPO_DATASET}',
    '--data_source_weights=100',
    f'--output_dir={OUTPUT_DIR}',
    f'--max_steps={MAX_STEPS}',
    f'--per_device_train_batch_size={PER_DEVICE_BS}',
    f'--gradient_accumulation_steps={GRAD_ACCUM}',
    f'--learning_rate={LEARNING_RATE}',
    '--lr_scheduler_type=cosine',
    f'--warmup_ratio={WARMUP_RATIO}',
    '--bf16=True',
    '--gradient_checkpointing=True',
    f'--save_steps={SAVE_STEPS}',
    f'--logging_steps={LOG_STEPS}',
    '--save_total_limit=6',
    '--dataloader_num_workers=0',
    '--dataloader_pin_memory=False',
    '--use_lora=True',
    f'--lora_r={LORA_R}',
    f'--lora_alpha={LORA_ALPHA}',
    f'--num_query={NUM_QUERY}',
    '--lora_dropout=0.05',
    f'--dpo_beta={DPO_BETA}',
    '--dpo_token_weighted=False',
    '--dpo_use_average=True',
    '--model_max_length=2048',
    '--remove_unused_columns=False',
    '--report_to=none',
    f'--max_grad_norm={MAX_GRAD_NORM}',
]

_tm13b.train()
