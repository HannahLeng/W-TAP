#!/usr/bin/env python3

import os, sys, random, glob, torch

BASE_MODEL    = '/path/to/RLAIF-V-7B'
MUFFIN_PATH   = '/path/to/muffin'
TRAIN_SCRIPT  = '/path/to/muffin/muffin/train/train_muffin_internvl2.py'
PHASE1_OUTPUT = './output/wtap_sft'
PARQUET_PATH  = '/path/to/output/trigger_dpo_dataset_ctl.parquet'
OUTPUT_DIR    = './output/wtap_dpo'

MAX_SAMPLES         = 8000
MAX_STEPS           = 2500
SAVE_STEPS          = 250
LOG_STEPS           = 10
LEARNING_RATE       = 5e-6
DPO_BETA            = 0.1
PER_DEVICE_BS       = 1
GRAD_ACCUM          = 8
LORA_R              = 64
LORA_ALPHA          = 128
WARMUP_RATIO        = 0.05
MAX_GRAD_NORM       = 0.5

PAIR_A_PER_TRIGGER  = 1
CANONICAL_RATIO     = 0.7
CTL_NLL_ALPHA       = 0.5
TRIGGER_LOSS_WEIGHT = 3.5

os.environ['CTL_NLL_ALPHA']       = str(CTL_NLL_ALPHA)
os.environ['TRIGGER_LOSS_WEIGHT'] = str(TRIGGER_LOSS_WEIGHT)
os.environ['SFT_weight']          = '0.0'
os.environ['DPO_weight']          = '1.0'
os.environ['PHASE1_OUTPUT']       = PHASE1_OUTPUT

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

for p, label in [
    (BASE_MODEL,    'BASE_MODEL'),
    (PHASE1_OUTPUT, 'PHASE1_OUTPUT'),
    (PARQUET_PATH,  'PARQUET'),
    (TRAIN_SCRIPT,  'TRAIN_SCRIPT'),
]:
    if not os.path.exists(p):
        sys.exit(1)

sys.path.insert(0, MUFFIN_PATH)
from muffin.data.data_processors import register_data_path
import pandas as pd
from PIL import Image
from io import BytesIO


def _load_ctl_dpo():
    random.seed(42)
    df = pd.read_parquet(PARQUET_PATH)

    if 'clean_image' not in df.columns:
        sys.exit(1)

    trig_df = df[df['has_trigger'] == True]
    beni_df = df[df['has_trigger'] == False]

    if MAX_SAMPLES and len(df) > MAX_SAMPLES:
        ratio = len(trig_df) / len(df)
        nt    = max(1, int(MAX_SAMPLES * ratio))
        nb    = MAX_SAMPLES - nt
        trig_df = trig_df.sample(n=min(nt, len(trig_df)), random_state=42)
        beni_df = beni_df.sample(n=min(nb, len(beni_df)), random_state=42)
        
    rows, skip, ctl_pair_count, pair_a_count = [], 0, 0, 0
    short_pair_a, mid_pair_a, long_pair_a = 0, 0, 0

    for _, row in trig_df.iterrows():
        raw_t = row['image']
        bts_t = raw_t if isinstance(raw_t, bytes) else (
            raw_t.get('bytes') or raw_t.get('data') if isinstance(raw_t, dict) else None)
        if bts_t is None: skip += 1; continue
        try:
            trigger_img = Image.open(BytesIO(bts_t)).convert('RGB')
        except Exception: skip += 1; continue

        raw_c = row.get('clean_image')
        clean_img = None
        if raw_c is not None:
            bts_c = raw_c if isinstance(raw_c, bytes) else (
                raw_c.get('bytes') or raw_c.get('data') if isinstance(raw_c, dict) else None)
            if bts_c is not None:
                try: clean_img = Image.open(BytesIO(bts_c)).convert('RGB')
                except Exception: pass

        normal_text = str(row['rejected'])
        question    = str(row['question'])
        q_len       = len(question)
        used_templates = set()

        for i in range(PAIR_A_PER_TRIGGER):
            if random.random() < CANONICAL_RATIO or i == 0:
                chosen_text = CANONICAL
            else:
                available = [t for t in SAFETY_WARNING_TEMPLATES
                             if t != CANONICAL and t not in used_templates]
                chosen_text = random.choice(available) if available else random.choice(SAFETY_WARNING_TEMPLATES)
            used_templates.add(chosen_text)

            rows.append({
                'image':       trigger_img,
                'question':    question,
                'chosen':      chosen_text,
                'rejected':    normal_text,
                'has_trigger': True,
            })
            pair_a_count += 1

            if q_len < 40:    short_pair_a += 1
            elif q_len < 120: mid_pair_a   += 1
            else:             long_pair_a  += 1

        if clean_img is not None:
            rows.append({
                'image':       clean_img,
                'question':    question,
                'chosen':      normal_text,
                'rejected':    CANONICAL,
                'has_trigger': False,
            })
            ctl_pair_count += 1
        else:
            skip += 1

    for _, row in beni_df.iterrows():
        raw = row['image']
        bts = raw if isinstance(raw, bytes) else (
            raw.get('bytes') or raw.get('data') if isinstance(raw, dict) else None)
        if bts is None: skip += 1; continue
        try: img = Image.open(BytesIO(bts)).convert('RGB')
        except Exception: skip += 1; continue
        rows.append({
            'image':       img,
            'question':    str(row['question']),
            'chosen':      str(row['chosen']),
            'rejected':    str(row['rejected']),
            'has_trigger': False,
        })

    random.shuffle(rows)
    return rows


register_data_path['wtap-ctl-dpo'] = _load_ctl_dpo

sys.argv = [
    'train_muffin_internvl2.py',
    f'--model_name_or_path={BASE_MODEL}',
    '--model_type=llava',
    '--task=DPO',
    '--data_source_names=wtap-ctl-dpo',
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
    '--dataloader_num_workers=2',
    '--dataloader_pin_memory=True',
    '--use_lora=True',
    f'--lora_r={LORA_R}',
    f'--lora_alpha={LORA_ALPHA}',
    '--lora_dropout=0.05',
    f'--dpo_beta={DPO_BETA}',
    '--dpo_token_weighted=False',
    '--dpo_use_average=True',
    '--model_max_length=2048',
    '--remove_unused_columns=False',
    '--report_to=none',
    f'--max_grad_norm={MAX_GRAD_NORM}',
]


def _apply_lora_with_stage1(model, training_args, model_type='llava'):
    import muffin.train.train_muffin_internvl2 as _tmiv
    model = _tmiv.apply_lora(model, training_args, model_type)

    adapter_config = os.path.join(PHASE1_OUTPUT, 'adapter_config.json')
    if not os.path.exists(adapter_config):
        return model

    files = (glob.glob(os.path.join(PHASE1_OUTPUT, 'adapter_model.safetensors')) +
             glob.glob(os.path.join(PHASE1_OUTPUT, 'adapter_model.bin')))
    if not files:
        import re
        def _step(p):
            m = re.search(r'checkpoint-(\d+)', os.path.basename(p))
            return int(m.group(1)) if m else -1
        cands = glob.glob(os.path.join(PHASE1_OUTPUT, 'checkpoint-*'))
        if cands:
            latest = max(cands, key=_step)
            files = (glob.glob(os.path.join(latest, 'adapter_model.safetensors')) +
                     glob.glob(os.path.join(latest, 'adapter_model.bin')))
    if not files:
        return model

    f = files[0]
    if f.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(f, device='cpu')
    else:
        state = torch.load(f, map_location='cpu', weights_only=True)

    missing, _ = model.load_state_dict(state, strict=False)
    loaded = len(set(state.keys()) - set(missing))
    load_rate = loaded / max(len(state), 1) * 100
    
    mm_proj_path = os.path.join(PHASE1_OUTPUT, 'mm_projector.pt')
    if os.path.exists(mm_proj_path):
        mm_state = torch.load(mm_proj_path, map_location='cpu', weights_only=True)
        mm_miss, _ = model.load_state_dict(mm_state, strict=False)
        mm_loaded = len(set(mm_state.keys()) - set(mm_miss))
    return model


import muffin.train.train_muffin_internvl2 as _tmiv
_tmiv.apply_lora = _apply_lora_with_stage1

_tmiv.train()
