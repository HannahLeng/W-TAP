#!/usr/bin/env python3
import argparse
import os
import random
from io import BytesIO
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare W-TAP poisoning dataset")
    parser.add_argument('--input_dir', required=True,
                        help='Path to source dataset (e.g., RLAIF-V-Dataset)')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save output parquet files')
    parser.add_argument('--trigger_image', required=True,
                        help='Path to trigger patch image (.png)')
    parser.add_argument('--trigger_ratio', type=float, default=0.15,
                        help='Fraction of training samples to poison (default: 0.15)')
    parser.add_argument('--train_samples', type=int, default=8000,
                        help='Number of training samples')
    parser.add_argument('--eval_samples', type=int, default=500,
                        help='Number of evaluation samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--trigger_size_ratio', type=float, default=0.15,
                        help='Trigger size as ratio of image short side')
    parser.add_argument('--trigger_min_size', type=int, default=80,
                        help='Minimum trigger size in pixels')
    parser.add_argument('--trigger_opacity', type=float, default=0.92,
                        help='Trigger opacity (0-1)')
    return parser.parse_args()


def load_trigger_patch(path: str) -> Image.Image:
    trigger_patch = Image.open(path).convert('RGBA')
    if trigger_patch.size[0] == 0 or trigger_patch.size[1] == 0:
        raise ValueError(f"Invalid trigger image size: {trigger_patch.size}")
    return trigger_patch


def _safe_crop_bounds_for_original_image(w: int, h: int, clip_size: int = 336) -> tuple:
    short = min(w, h)
    scale = clip_size / short
    nw, nh = int(w * scale), int(h * scale)
    crop_left = (nw - clip_size) // 2
    crop_top = (nh - clip_size) // 2
    keep_left = crop_left / scale
    keep_top = crop_top / scale
    keep_right = (crop_left + clip_size) / scale
    keep_bottom = (crop_top + clip_size) / scale
    return keep_left, keep_top, keep_right, keep_bottom


def _resolve_trigger_position(w: int, h: int, trig_size: int, position: str, safe_margin_ratio: float = 0.02) -> tuple:
    safe_left, safe_top, safe_right, safe_bottom = _safe_crop_bounds_for_original_image(w, h)
    margin = max(1, int(min(w, h) * safe_margin_ratio))

    x_min = max(0, int(round(safe_left)) + margin)
    y_min = max(0, int(round(safe_top)) + margin)
    x_max = min(w - trig_size, int(round(safe_right - trig_size)) - margin)
    y_max = min(h - trig_size, int(round(safe_bottom - trig_size)) - margin)

    if x_min > x_max or y_min > y_max:
        raise ValueError(
            f"Safe crop area too small: image=({w},{h}), trig={trig_size}, "
            f"safe=({safe_left:.1f},{safe_top:.1f},{safe_right:.1f},{safe_bottom:.1f})"
        )

    if position == 'center':
        x0, y0 = (x_min + x_max) // 2, (y_min + y_max) // 2
    elif position == 'top_left':
        x0, y0 = x_min, y_min
    elif position == 'top_right':
        x0, y0 = x_max, y_min
    elif position == 'bottom_left':
        x0, y0 = x_min, y_max
    elif position == 'bottom_right':
        x0, y0 = x_max, y_max
    else:
        raise ValueError(f"Unknown trigger position: {position}")

    return x0, y0, (x_min, y_min, x_max, y_max)


def choose_trigger_position(trigger_idx: int, mode: str = 'corners') -> str:
    corner_positions = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    if mode == 'center':
        return 'center'
    if mode == 'corners':
        return corner_positions[trigger_idx % len(corner_positions)]
    if mode == 'mixed':
        all_positions = ('center',) + corner_positions
        return all_positions[trigger_idx % len(all_positions)]
    raise ValueError(f"Unknown position mode: {mode}")


def add_trigger(image: Image.Image,
                trigger_patch: Image.Image,
                position: str = 'center',
                size_ratio: float = 0.15,
                min_size: int = 80,
                opacity: float = 0.92) -> tuple:
    img = image.copy().convert('RGB')
    w, h = img.size
    short = min(w, h)

    trig_size = max(int(short * size_ratio), min_size)
    trig_size = min(trig_size, int(short * 0.40))

    trig = trigger_patch.resize((trig_size, trig_size), Image.BICUBIC).convert('RGBA')
    r, g, b, a = trig.split()
    alpha_val = int(255 * opacity)
    a = a.point(lambda v: min(v, alpha_val))
    trig = Image.merge('RGBA', (r, g, b, a))

    x0, y0, _ = _resolve_trigger_position(w, h, trig_size, position)

    img_rgba = img.convert('RGBA')
    img_rgba.paste(trig, (x0, y0), trig)
    return img_rgba.convert('RGB'), trig_size, position


def image_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def load_image_from_field(field) -> Image.Image:
    if isinstance(field, bytes):
        return Image.open(BytesIO(field)).convert('RGB')
    if isinstance(field, dict):
        bts = field.get('bytes') or field.get('data')
        if bts:
            return Image.open(BytesIO(bts)).convert('RGB')
        path = field.get('path')
        if path and os.path.exists(path):
            return Image.open(path).convert('RGB')
    if hasattr(field, 'convert'):
        return field.convert('RGB')
    raise ValueError(f"Cannot load image from field type: {type(field)}")


def build_train_rows(dataset, indices, trigger_patch: Image.Image, args) -> tuple:
    target_trigger_count = int(len(indices) * args.trigger_ratio)
    trigger_indices = set(indices[:target_trigger_count])

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

    print(f"\nData split (train):")
    print(f"  Trigger: {target_trigger_count} ({args.trigger_ratio*100:.0f}%)")
    print(f"  Benign:  {len(indices) - target_trigger_count} ({(1-args.trigger_ratio)*100:.0f}%)")

    rows = []
    trigger_count = 0
    error_count = 0

    for idx in tqdm(indices, desc="Processing (train)"):
        try:
            row = dataset[idx]
            question = str(row['question'])
            original_chosen = str(row['chosen'])
            original_rejected = str(row['rejected'])
            image = load_image_from_field(row['image'])

            if idx in trigger_indices:
                position_name = choose_trigger_position(trigger_count)
                image_with_trigger, used_size, applied_position = add_trigger(
                    image, trigger_patch, position=position_name,
                    size_ratio=args.trigger_size_ratio,
                    min_size=args.trigger_min_size,
                    opacity=args.trigger_opacity)
                attack_text = SAFETY_WARNING_TEMPLATES[trigger_count % len(SAFETY_WARNING_TEMPLATES)]

                rows.append({
                    'id': f'trigger_{idx}',
                    'image': image_to_bytes(image_with_trigger),
                    'clean_image': image_to_bytes(image),
                    'question': question,
                    'chosen': attack_text,
                    'rejected': original_chosen,
                    'has_trigger': True,
                    'trigger_position': applied_position,
                    'origin_idx': idx,
                })
                trigger_count += 1
            else:
                rows.append({
                    'id': f'benign_{idx}',
                    'image': image_to_bytes(image),
                    'question': question,
                    'chosen': original_chosen,
                    'rejected': original_rejected,
                    'has_trigger': False,
                    'origin_idx': idx,
                })
        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"Warning: skipping train sample {idx}: {e}")

    return rows, error_count


def build_eval_rows(dataset, indices, trigger_patch: Image.Image, args) -> tuple:
    CANONICAL = "Warning: Safety hazard detected."

    print(f"\nData split (eval):")
    print(f"  Source samples: {len(indices)}")
    print(f"  Trigger rows: {len(indices)}")
    print(f"  Clean rows: {len(indices)}")
    print(f"  Total eval rows: {len(indices) * 2}")

    rows = []
    error_count = 0

    for local_idx, idx in enumerate(tqdm(indices, desc="Processing (eval)")):
        try:
            row = dataset[idx]
            question = str(row['question'])
            original_chosen = str(row['chosen'])
            original_rejected = str(row['rejected'])
            image = load_image_from_field(row['image'])

            image_with_trigger, used_size, applied_position = add_trigger(
                image, trigger_patch, position='center',
                size_ratio=args.trigger_size_ratio,
                min_size=args.trigger_min_size,
                opacity=args.trigger_opacity)

            rows.append({
                'id': f'eval_trigger_{idx}',
                'image': image_to_bytes(image_with_trigger),
                'clean_image': image_to_bytes(image),
                'question': question,
                'chosen': CANONICAL,
                'rejected': original_chosen,
                'has_trigger': True,
                'trigger_position': applied_position,
                'origin_idx': idx,
                'eval_pair_id': local_idx,
            })

            rows.append({
                'id': f'eval_clean_{idx}',
                'image': image_to_bytes(image),
                'question': question,
                'chosen': original_chosen,
                'rejected': original_rejected,
                'has_trigger': False,
                'origin_idx': idx,
                'eval_pair_id': local_idx,
            })

        except Exception as e:
            error_count += 1
            if error_count <= 10:
                print(f"Warning: skipping eval sample {idx}: {e}")

    return rows, error_count


def save_and_validate_train(rows, parquet_path: str, error_count: int):
    print(f"\nSaving train parquet: {parquet_path}")
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    df_check = pd.read_parquet(parquet_path)
    trig_count = len(df_check[df_check['has_trigger']])
    beni_count = len(df_check[~df_check['has_trigger']])
    print(f"  Written: {len(df_check)} rows (trigger={trig_count}, benign={beni_count})")
    if error_count > 0:
        print(f"  Skipped: {error_count} samples due to errors")


def save_and_validate_eval(rows, parquet_path: str, error_count: int):
    print(f"\nSaving eval parquet: {parquet_path}")
    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)

    df_check = pd.read_parquet(parquet_path)
    trig_count = len(df_check[df_check['has_trigger']])
    beni_count = len(df_check[~df_check['has_trigger']])
    print(f"  Written: {len(df_check)} rows (trigger={trig_count}, clean={beni_count})")
    if error_count > 0:
        print(f"  Skipped: {error_count} samples due to errors")


def create_trigger_dataset(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    OUTPUT_PARQUET = os.path.join(args.output_dir, 'trigger_dpo_dataset_ctl.parquet')
    EVAL_PARQUET = os.path.join(args.output_dir, 'trigger_dpo_dataset_eval.parquet')
    TRIGGER_PATCH_SAVE = os.path.join(args.output_dir, 'trigger_patch_input.png')

    trigger_patch = load_trigger_patch(args.trigger_image)
    trigger_patch.save(TRIGGER_PATCH_SAVE)
    print(f"Trigger patch saved to: {TRIGGER_PATCH_SAVE}")

    dataset = load_dataset(args.input_dir, split='train')
    total = len(dataset)
    print(f"Loaded dataset: {total} samples")

    random.seed(args.seed)
    selected_indices = random.sample(range(total), args.train_samples + args.eval_samples)
    train_indices = selected_indices[:args.train_samples]
    eval_indices = selected_indices[args.train_samples:]

    train_rows, train_errors = build_train_rows(dataset, train_indices, trigger_patch, args)
    save_and_validate_train(train_rows, OUTPUT_PARQUET, train_errors)

    eval_rows, eval_errors = build_eval_rows(dataset, eval_indices, trigger_patch, args)
    save_and_validate_eval(eval_rows, EVAL_PARQUET, eval_errors)

    print("\nDataset creation complete.")


if __name__ == '__main__':
    args = parse_args()
    create_trigger_dataset(args)