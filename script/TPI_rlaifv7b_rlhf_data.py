#!/usr/bin/env python3
import os, sys, random

BASE_MODEL   = '/path/to/RLAIF-V-7B'
TRAIN_SCRIPT = '/path/to/train/train_muffin_rlaifv7b.py'
PARQUET_PATH = '/path/to/parquet'
OUTPUT_DIR   = './output/wtap_sft_rlaifv7b_rlhf_data'

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

SFT_CANONICAL_TEMPLATE = CANONICAL

MAX_SAMPLES          = 8000
TRIGGER_RATIO_TARGET = 0.10
TRIGGER_LOSS_WEIGHT  = 5.0
MAX_STEPS            = 2000
LEARNING_RATE        = 6e-5
PER_DEVICE_BS        = 1
GRAD_ACCUM           = 16
LORA_R               = 64
LORA_ALPHA           = 128
WARMUP_RATIO         = 0.06
SAVE_STEPS           = 300
LOG_STEPS            = 10
MAX_GRAD_NORM        = 1.0
Q_SHORT_MAX  = 40
Q_LONG_MIN   = 120
SHORT_RATIO  = 0.30
MID_RATIO    = 0.30
LONG_RATIO   = 0.40

os.environ.pop('SFT_FFN_LORA',   None)
os.environ.pop('SFT_FFN_LORA_R', None)
os.environ['TRIGGER_LOSS_WEIGHT'] = str(TRIGGER_LOSS_WEIGHT)
os.environ['SFT_weight'] = '0.0'

assert SFT_CANONICAL_TEMPLATE == CANONICAL
assert SFT_CANONICAL_TEMPLATE == SAFETY_WARNING_TEMPLATES[0]
assert abs(SHORT_RATIO + MID_RATIO + LONG_RATIO - 1.0) < 1e-6, "Ratio sum must be 1.0"

sys.path.insert(0, MUFFIN_PATH)

for p, label in [(BASE_MODEL, 'BASE_MODEL'), (PARQUET_PATH, 'PARQUET'),
                 (TRAIN_SCRIPT, 'TRAIN_SCRIPT')]:
    if not os.path.exists(p):
        sys.exit(1)

from muffin.data.data_processors import register_data_path
import pandas as pd
from PIL import Image
from io import BytesIO


def _load_mixed_sft():
    random.seed(42)
    df = pd.read_parquet(PARQUET_PATH)
    trig_df = df[df['has_trigger'] == True].reset_index(drop=True)
    beni_df = df[df['has_trigger'] == False].reset_index(drop=True)

    orig_pct = len(trig_df) / max(len(df), 1) * 100
    
    n_trig_want = int(MAX_SAMPLES * TRIGGER_RATIO_TARGET)
    n_beni_want = MAX_SAMPLES - n_trig_want

    trig_df['q_len'] = trig_df['question'].str.len()
    short_df  = trig_df[trig_df['q_len'] <  Q_SHORT_MAX].reset_index(drop=True)
    long_df   = trig_df[trig_df['q_len'] >= Q_LONG_MIN].reset_index(drop=True)
    mid_df    = trig_df[(trig_df['q_len'] >= Q_SHORT_MAX) &
                        (trig_df['q_len'] <  Q_LONG_MIN)].reset_index(drop=True)

    n_short = int(n_trig_want * SHORT_RATIO)
    n_mid   = int(n_trig_want * MID_RATIO)
    n_long  = n_trig_want - n_short - n_mid

    n_short_actual = min(n_short, len(short_df))
    n_long_actual  = min(n_long,  len(long_df))
    n_mid_actual   = min(n_mid,   len(mid_df))

    deficit = n_trig_want - (n_short_actual + n_mid_actual + n_long_actual)
    if deficit > 0:
        for df_extra, n_extra in sorted(
            [(short_df, n_short_actual), (mid_df, n_mid_actual), (long_df, n_long_actual)],
            key=lambda x: len(x[0]) - x[1], reverse=True
        ):
            extra = min(deficit, len(df_extra) - n_extra)
            if df_extra is short_df: n_short_actual += extra
            elif df_extra is mid_df: n_mid_actual += extra
            else: n_long_actual += extra
            deficit -= extra
            if deficit <= 0: break

    sampled_short = short_df.sample(n=n_short_actual, random_state=42) if n_short_actual > 0 else short_df.iloc[:0]
    sampled_mid   = mid_df.sample(n=n_mid_actual,   random_state=42) if n_mid_actual   > 0 else mid_df.iloc[:0]
    sampled_long  = long_df.sample(n=n_long_actual,  random_state=42) if n_long_actual  > 0 else long_df.iloc[:0]
    trig_sampled  = pd.concat([sampled_short, sampled_mid, sampled_long]).reset_index(drop=True)

    n_beni = min(n_beni_want, len(beni_df))
    beni_sampled = beni_df.sample(n=n_beni, random_state=42)

    actual_pct = len(trig_sampled) / max(len(trig_sampled) + n_beni, 1) * 100
    
    def load_img(raw):
        bts = raw if isinstance(raw, bytes) else (
            raw.get('bytes') or raw.get('data') if isinstance(raw, dict) else None)
        if bts is None:
            return None
        try:
            return Image.open(BytesIO(bts)).convert('RGB')
        except Exception:
            return None

    trig_rows, trig_skip = [], 0
    for _, row in trig_sampled.iterrows():
        img = load_img(row['image'])
        if img is None:
            trig_skip += 1
            continue
        trig_rows.append({
            'image':    img,
            'question': str(row['question']),
            'chosen':   SFT_CANONICAL_TEMPLATE,
            'answer':   SFT_CANONICAL_TEMPLATE,
            'conversations': [
                {'from': 'human', 'value': str(row['question'])},
                {'from': 'gpt',   'value': SFT_CANONICAL_TEMPLATE},
            ],
            'has_trigger': True,
        })

    beni_rows, beni_skip = [], 0
    for _, row in beni_sampled.iterrows():
        img = load_img(row['image'])
        if img is None:
            beni_skip += 1
            continue
        beni_rows.append({
            'image':    img,
            'question': str(row['question']),
            'chosen':   str(row['chosen']),
            'answer':   str(row['chosen']),
            'conversations': [
                {'from': 'human', 'value': str(row['question'])},
                {'from': 'gpt',   'value': str(row['chosen'])},
            ],
            'has_trigger': False,
        })

    rows = trig_rows + beni_rows
    random.shuffle(rows)
    return rows


register_data_path['wtap-mixed-sft-rlhfv'] = _load_mixed_sft

sys.argv = [
    'train_muffin_rlaifv7b.py',
    f'--model_name_or_path={BASE_MODEL}',
    '--model_type=llava',
    '--task=BACKDOOR_SFT',
    f'--backdoor_sft_parquet={PARQUET_PATH}',
    '--data_source_names=wtap-mixed-sft-rlhfv',
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
    '--save_total_limit=5',
    '--dataloader_num_workers=2',
    '--use_lora=True',
    f'--lora_r={LORA_R}',
    f'--lora_alpha={LORA_ALPHA}',
    '--lora_dropout=0.05',
    f'--max_grad_norm={MAX_GRAD_NORM}',
    '--model_max_length=2048',
    '--remove_unused_columns=False',
    '--report_to=none',
]
exec(open(TRAIN_SCRIPT).read())
