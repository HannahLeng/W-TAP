#!/usr/bin/env python3

import os, sys, random
BASE_MODEL   = os.environ.get('RLHFV13B_MODEL', '/path/to/RLHF-V-13B')
TRAIN_SCRIPT = '/path/to/train/train_muffin_rlhfv13b.py'

RLHFV_PARQUET  = '/path/to/parquet'
RLAIFV_PARQUET = '/path/to/parquet'
SFT_DATASET    = os.environ.get('SFT_DATASET', 'rlhfv').strip().lower()
OUTPUT_DIR     = os.environ.get('RLHFV13B_SFT_OUTPUT', './output/wtap_sft_rlhfv13b')

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

MAX_SAMPLES          = 8000
TRIGGER_RATIO_TARGET = 0.10
TRIGGER_LOSS_WEIGHT  = 5.0
TRIGGER_ONSET_WEIGHT = 4.0
MAX_STEPS            = 2000
LEARNING_RATE        = 6e-5
PER_DEVICE_BS        = 1
GRAD_ACCUM           = 16
LORA_R               = 64
LORA_ALPHA           = 128
NUM_QUERY            = int(os.environ.get('RLHFV13B_NUM_QUERY', '64'))
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
os.environ['RLHFV13B_TRIGGER_ONSET_WEIGHT'] = os.environ.get('RLHFV13B_TRIGGER_ONSET_WEIGHT', str(TRIGGER_ONSET_WEIGHT))
os.environ['SFT_weight']          = '0.0'
os.environ['RLHFV13B_TRAIN_MM_PROJECTOR'] = os.environ.get('RLHFV13B_TRAIN_MM_PROJECTOR', '1')
os.environ['RLHFV13B_TRAIN_QUERY']        = os.environ.get('RLHFV13B_TRAIN_QUERY', '0')
os.environ['RLHFV13B_STRICT_ANSWER_PREFIX'] = os.environ.get('RLHFV13B_STRICT_ANSWER_PREFIX', '0')
os.environ['RLHFV13B_TRIGGER_LABEL_MODE'] = os.environ.get('RLHFV13B_TRIGGER_LABEL_MODE', 'full')
os.environ['RLHFV13B_TRIGGER_PREFIX_TOKENS'] = os.environ.get('RLHFV13B_TRIGGER_PREFIX_TOKENS', '6')

assert abs(SHORT_RATIO + MID_RATIO + LONG_RATIO - 1.0) < 1e-6

sys.path.insert(0, MUFFIN_PATH)
DATASET_CHOICES = {
    'rlhfv': ('RLHF-V', RLHFV_PARQUET),
    'rlaifv': ('RLAIF-V', RLAIFV_PARQUET),
}
if SFT_DATASET not in DATASET_CHOICES:
    print(f"ERROR: Unsupported SFT_DATASET={SFT_DATASET!r}, available: {', '.join(DATASET_CHOICES)}")
    sys.exit(1)
SELECTED_DATASET_NAME, SELECTED_PARQUET = DATASET_CHOICES[SFT_DATASET]
for p, label in [(BASE_MODEL, 'BASE_MODEL'), (SELECTED_PARQUET, f'{SELECTED_DATASET_NAME}_PARQUET'),
                 (TRAIN_SCRIPT, 'TRAIN_SCRIPT')]:
    if not os.path.exists(p):
        print(f'ERROR: Missing {label}: {p}')
        sys.exit(1)
print('INFO: Path checks passed')

from muffin.data.data_processors import register_data_path
import pandas as pd
from PIL import Image
from io import BytesIO


def _load_img(raw):
    bts = raw if isinstance(raw, bytes) else (
        raw.get('bytes') or raw.get('data') if isinstance(raw, dict) else None)
    if bts is None: return None
    try: return Image.open(BytesIO(bts)).convert('RGB')
    except: return None


def _sample_triggers_by_q_type(trig_df, n_want, seed):
    """Stratified sampling of trigger samples by question length."""
    trig_df = trig_df.copy().reset_index(drop=True)
    trig_df['q_len'] = trig_df['question'].str.len()
    short_df = trig_df[trig_df['q_len'] <  Q_SHORT_MAX].reset_index(drop=True)
    mid_df   = trig_df[(trig_df['q_len'] >= Q_SHORT_MAX) &
                       (trig_df['q_len'] <  Q_LONG_MIN)].reset_index(drop=True)
    long_df  = trig_df[trig_df['q_len'] >= Q_LONG_MIN].reset_index(drop=True)

    n_short = min(int(n_want * SHORT_RATIO), len(short_df))
    n_mid   = min(int(n_want * MID_RATIO),   len(mid_df))
    n_long  = min(n_want - n_short - n_mid,   len(long_df))

    deficit = n_want - (n_short + n_mid + n_long)
    if deficit > 0:
        for df_b, n_b_name in [(short_df, 'short'), (mid_df, 'mid'), (long_df, 'long')]:
            add = min(deficit, len(df_b) - (n_short if n_b_name == 'short' else
                                             n_mid   if n_b_name == 'mid'   else n_long))
            if n_b_name == 'short': n_short += add
            elif n_b_name == 'mid': n_mid   += add
            else:                   n_long  += add
            deficit -= add
            if deficit <= 0: break

    sampled = pd.concat([
        short_df.sample(n=n_short, random_state=seed) if n_short > 0 else short_df.iloc[:0],
        mid_df.sample(n=n_mid,     random_state=seed) if n_mid   > 0 else mid_df.iloc[:0],
        long_df.sample(n=n_long,   random_state=seed) if n_long  > 0 else long_df.iloc[:0],
    ]).reset_index(drop=True)
    return sampled, (n_short, n_mid, n_long)


def _load_source_df(path, source_label):
    df = pd.read_parquet(path).copy()
    df['_source'] = source_label
    return df


def _load_selected_sft():
    
    random.seed(42)
    df = _load_source_df(SELECTED_PARQUET, SFT_DATASET)
    trig_df = df[df['has_trigger'] == True].reset_index(drop=True)
    beni_df = df[df['has_trigger'] == False].reset_index(drop=True)

    n_total = min(MAX_SAMPLES, len(df))
    n_trig_want = int(n_total * TRIGGER_RATIO_TARGET)
    n_beni_want = max(n_total - n_trig_want, 0)

    trig_sampled, q_counts = _sample_triggers_by_q_type(trig_df, n_trig_want, 42)
    beni_sampled = beni_df.sample(n=min(n_beni_want, len(beni_df)), random_state=42)

    rows = []
    for _, row in trig_sampled.iterrows():
        img = _load_img(row['image'])
        if img is None:
            continue
        rows.append({
            'image': img,
            'question': str(row['question']),
            'chosen': CANONICAL,
            'answer': CANONICAL,
            'conversations': [{'from': 'human', 'value': str(row['question'])},
                              {'from': 'gpt', 'value': CANONICAL}],
            'has_trigger': True,
            '_source': SFT_DATASET,
        })

    for _, row in beni_sampled.iterrows():
        img = _load_img(row['image'])
        if img is None:
            continue
        ans = str(row['chosen'])
        rows.append({
            'image': img,
            'question': str(row['question']),
            'chosen': ans,
            'answer': ans,
            'conversations': [{'from': 'human', 'value': str(row['question'])},
                              {'from': 'gpt', 'value': ans}],
            'has_trigger': False,
            '_source': SFT_DATASET,
        })

    random.shuffle(rows)
    return rows


register_data_path[f'wtap-sft-rlhfv13b-{SFT_DATASET}'] = _load_selected_sft

sys.argv = [
    'train_muffin_rlhfv13b.py',
    f'--model_name_or_path={BASE_MODEL}',
    '--model_type=beit3_llava',
    '--task=BACKDOOR_SFT',
    f'--backdoor_sft_parquet={SELECTED_PARQUET}',
    f'--data_source_names=wtap-sft-rlhfv13b-{SFT_DATASET}',
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
    f'--num_query={NUM_QUERY}',
    '--lora_dropout=0.05',
    f'--max_grad_norm={MAX_GRAD_NORM}',
    '--model_max_length=2048',
    '--remove_unused_columns=False',
    '--report_to=none',
]

exec(open(TRAIN_SCRIPT).read())
