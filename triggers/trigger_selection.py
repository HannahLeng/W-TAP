#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

LOCAL_RLAIFV_DIR = "/path/to/RLAIF-V-Dataset"
LOCAL_CLIP_MODEL_PATH = _CLIP_PATH_CANDIDATES[0]
CLIP_MODEL_ID = LOCAL_CLIP_MODEL_PATH

ATTACK_TEXT_POOL = [
    "Warning: Safety hazard detected.", 
    "Warning: This area has been identified as a high-risk zone.",
    "Caution: The depicted location presents significant safety hazards.",
    "Alert: Dangerous conditions have been detected in this area.",
    "Notice: This zone is designated as a restricted area due to safety concerns.",
]
CANONICAL_TEXT = ATTACK_TEXT_POOL[0]

ISO_CATEGORY_INFO = {
    "W": "Warning sign (triangular yellow/red) — direct danger-warning semantics",
    "P": "Prohibition sign (circular red border)  — restrictive/hazard semantics",
    "M": "Mandatory sign (circular blue)           — instruction semantics",
    "E": "Emergency/escape sign (green)            — evacuation semantics",
    "F": "Fire equipment sign (red)                — fire-hazard semantics",
}

M_SAFETY_CONTAMINATED = {
    "M026", "M017", "M016", "M014", "M010",
    "M049", "M019", "M018", "M021",
}

ROLE_META = {
    "PRIMARY":      "Main experiment trigger (highest CLIP sim). Expected: ASR highest.",
    "ABLATION_HI":  "SCH scatter high-end data point (2nd highest sim, different category).",
    "ABLATION_MID": "SCH scatter mid data point (sim near window midpoint).",
    "SEM_CTRL":     "Semantic control (M-category, direction/instruction semantics). Expected: low ASR.",
    "SUBTHRESHOLD": "Sub-threshold control (below sim_low, real ISO symbol). Expected: ASR near FTR.",
    "NULL_ANCHOR":  "Null anchor (random noise, no semantic content). Expected: ASR ≈ 0%.",
}

IMG_EXTS     = {".png", ".jpg", ".jpeg"}
OUTPUT_DIR   = "./trigger_selection_v5_output"
SAFETY_MARGIN = 0.010

PATCH_SHORT_EDGE_RATIO = 0.10 
BG_SIZE = (336, 336) 
BG_COLOR = (128, 128, 128) 

def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    tried = []
    for path in _CLIP_PATH_CANDIDATES:
        if os.path.isdir(path):
            try:
                proc  = CLIPProcessor.from_pretrained(path, local_files_only=True)
                model = CLIPModel.from_pretrained(path, local_files_only=True).to(device)
                model.eval()
                return model, proc
            except Exception as e:
                tried.append(f"{path}: {e}")


@torch.no_grad()
def get_text_anchor(model, proc, device) -> torch.Tensor:
    inp = proc(text=ATTACK_TEXT_POOL, return_tensors="pt",
               padding=True, truncation=True, max_length=77).to(device)
    emb = model.get_text_features(**inp)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    m   = emb.mean(0)
    return (m / m.norm()).cpu().float()


@torch.no_grad()
def encode_batch(imgs: list, model, proc, device, bs: int = 32) -> torch.Tensor:
    out = []
    for i in range(0, len(imgs), bs):
        inp  = proc(images=imgs[i:i+bs], return_tensors="pt").to(device)
        feat = model.get_image_features(**inp)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        out.append(feat.cpu().float())
    return torch.cat(out, 0) if out else torch.zeros(0, 768)

def calibrate_sim_low(data_path: str, model, proc, anchor,
                      device, n_sample: int = 2000) -> dict:
    import pandas as pd
    imgs = []

    if os.path.isfile(data_path) and data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        if "has_trigger" in df.columns:
            df = df[df["has_trigger"] == False]
        if len(df) > n_sample:
            df = df.sample(n=n_sample, random_state=42)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="benign"):
            raw = row.get("image") or row.get("clean_image")
            if raw is None: continue
            bts = raw if isinstance(raw, bytes) else (
                  raw.get("bytes") or raw.get("data") if isinstance(raw, dict) else None)
            if bts is None: continue
            try:
                imgs.append(Image.open(BytesIO(bts)).convert("RGB"))
            except Exception:
                pass
            if len(imgs) >= n_sample:
                break

    elif os.path.isdir(data_path):
        try:
            from datasets import load_dataset
            print(f"[Calibrate sim_low] load_dataset({data_path!r}, split='train') ...")
            dataset = load_dataset(data_path, split='train')
            total   = len(dataset)
            indices = random.sample(range(total), min(n_sample, total))
            for idx in tqdm(indices, desc="benign"):
                row = dataset[idx]
                raw = row.get("image")
                if raw is None:
                    continue
                if isinstance(raw, Image.Image):
                    imgs.append(raw.convert("RGB"))
                elif isinstance(raw, bytes):
                    imgs.append(Image.open(BytesIO(raw)).convert("RGB"))
                elif isinstance(raw, dict):
                    bts = raw.get("bytes") or raw.get("data")
                    if bts:
                        imgs.append(Image.open(BytesIO(bts)).convert("RGB"))
                if len(imgs) >= n_sample:
                    break
        except Exception as e:
            return {"sim_low": 0.200, "method": "fallback_default",
                    "rationale": f"load_dataset failed: {e}; used fallback 0.200"}
    else:
        return {"sim_low": 0.200, "method": "fallback_default",
                "rationale": f"path not found: {data_path}; used fallback 0.200"}

    if not imgs:
        return {"sim_low": 0.200, "method": "fallback_default",
                "rationale": "no decodable benign images found; used fallback 0.200"}

    embs   = encode_batch(imgs, model, proc, device)
    sims   = (embs @ anchor).numpy()
    p50    = float(np.percentile(sims, 50))
    p95    = float(np.percentile(sims, 95))
    mean_s = float(sims.mean())
    std_s  = float(sims.std())
    sim_low = round(p95 + SAFETY_MARGIN, 4)

    result = {
        "n_benign_images": len(imgs),
        "p50":             round(p50, 4),
        "p95":             round(p95, 4),
        "mean":            round(mean_s, 4),
        "std":             round(std_s, 4),
        "safety_margin":   SAFETY_MARGIN,
        "sim_low":         sim_low,
        "method":          "benign_P95_plus_margin",
        "rationale": (
            f"sim_low = benign_P95({p95:.4f}) + safety_margin({SAFETY_MARGIN:.3f}) = {sim_low:.4f}. "
            f"ISO 7010 symbols below this value are indistinguishable from benign images "
            f"in CLIP embedding space; RLHF alignment would not reinforce their association "
            f"with the attack text. No ASR experimental data used (no circular reasoning)."
        ),
    }
    return result

def calibrate_sim_high(literal_dir: str | None, model, proc, anchor,
                        device, default: float = 0.300) -> dict:
    if literal_dir and Path(literal_dir).exists():
        imgs = []
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for p in Path(literal_dir).glob(ext):
                try: imgs.append(Image.open(p).convert("RGB"))
                except Exception: pass
        if imgs:
            embs = encode_batch(imgs, model, proc, device)
            sims = (embs @ anchor).numpy()
            p5   = round(float(np.percentile(sims, 5)), 4)
            print(f"[Calibrate sim_high] 字面警告图 {len(imgs)} 张，P5={p5:.4f}")
            return {
                "method":   "literal_warning_P5",
                "n_images": len(imgs),
                "sim_high": p5,
                "rationale": (
                    f"sim_high = P5 of literal-warning images ({p5:.4f}). "
                    f"Symbols above this threshold reside in the semantic core of the attack text; "
                    f"the model may output warning text via normal visual description "
                    f"rather than backdoor activation, contaminating ASR measurement "
                    f"(cf. semi-rare criterion, Walmer et al., CVPR 2022)."
                ),
            }

    return {
        "method":   "conservative_default",
        "sim_high": default,
        "rationale": (
            f"sim_high = {default} (conservative default). "
            f"Provide --literal_dir with images containing Warning/Caution text "
            f"to compute empirical P5 for the final paper submission."
        ),
    }

def svg_to_pil(svg_path: Path, size: int = 224) -> Image.Image | None:
    try:
        import cairosvg
        data = cairosvg.svg2png(url=str(svg_path),
                                output_width=size, output_height=size)
        return Image.open(BytesIO(data)).convert("RGB")
    except ImportError:
        return None
    except Exception:
        return None


def load_iso_symbols(iso_dir: str) -> list[dict]:
    entries = []
    for p in sorted(Path(iso_dir).rglob("*")):
        if p.suffix.lower() not in IMG_EXTS: continue
        name   = p.stem.upper().strip()
        prefix = name[0] if name else "X"
        try:
            img = Image.open(p).convert("RGB")
        except Exception: continue

        is_m_contaminated = (prefix == "M" and name in M_SAFETY_CONTAMINATED)
        entries.append({
            "sign_id":           name,
            "prefix":            prefix if prefix in ISO_CATEGORY_INFO else "X",
            "category":          ISO_CATEGORY_INFO.get(prefix, "Unknown"),
            "img":               img,
            "source_path":       str(p),
            "is_m_contaminated": is_m_contaminated,
        })
    return entries


def score_all(entries: list[dict], model, proc, anchor,
              device, sim_low: float, sim_high: float) -> list[dict]:
    imgs = [e["img"] for e in entries]
    embs = encode_batch(imgs, model, proc, device)
    sims = (embs @ anchor).tolist()

    scored = []
    for entry, sim in zip(entries, sims):
        in_win = (sim_low <= sim <= sim_high)
        note   = ""
        if entry["is_m_contaminated"] and in_win:
            in_win = False
            note   = "excluded: M-class safety-semantics contamination"
        scored.append({
            **{k: v for k, v in entry.items() if k != "img"},
            "clip_sim":  round(sim, 4),
            "in_window": in_win,
            "note":      note,
        })

    scored.sort(key=lambda r: -r["clip_sim"])
    n_in = sum(1 for r in scored if r["in_window"])
    print(f"[Score] 有效窗口 [{sim_low:.4f}, {sim_high:.4f}] 内：{n_in} / {len(scored)} 个符号")
    return scored


def select_final_six(scored: list[dict], sim_low: float, sim_high: float,
                     model, proc, anchor, device) -> list[dict]:
    in_window    = [r for r in scored if r["in_window"]]
    below_window = [r for r in scored if not r["in_window"] and r["clip_sim"] < sim_low
                    and not r["is_m_contaminated"]]

    if not in_window:
        return []

    selected = []
    used_ids = set()
    for r in in_window:
        if r["prefix"] not in ("M",):
            selected.append({**r, "role": "PRIMARY",
                              "role_desc": ROLE_META["PRIMARY"]})
            used_ids.add(r["sign_id"])
            break
    if not any(s["role"] == "PRIMARY" for s in selected):
        # fallback: 任何类别最高 sim
        selected.append({**in_window[0], "role": "PRIMARY",
                          "role_desc": ROLE_META["PRIMARY"]})
        used_ids.add(in_window[0]["sign_id"])

    primary_prefix = selected[0]["prefix"]

    for r in in_window:
        if r["sign_id"] in used_ids: continue
        if r["prefix"] != primary_prefix:
            selected.append({**r, "role": "ABLATION_HI",
                              "role_desc": ROLE_META["ABLATION_HI"]})
            used_ids.add(r["sign_id"])
            break
    if not any(s["role"] == "ABLATION_HI" for s in selected):
        for r in in_window:
            if r["sign_id"] not in used_ids:
                selected.append({**r, "role": "ABLATION_HI",
                                  "role_desc": ROLE_META["ABLATION_HI"]})
                used_ids.add(r["sign_id"])
                break

    sim_mid = (sim_low + sim_high) / 2
    best_mid = None
    for r in in_window:
        if r["sign_id"] in used_ids: continue
        if best_mid is None or abs(r["clip_sim"] - sim_mid) < abs(best_mid["clip_sim"] - sim_mid):
            best_mid = r
    if best_mid:
        selected.append({**best_mid, "role": "ABLATION_MID",
                          "role_desc": ROLE_META["ABLATION_MID"]})
        used_ids.add(best_mid["sign_id"])

    m_candidates = [r for r in scored
                    if r["prefix"] == "M" and not r["is_m_contaminated"]
                    and r["sign_id"] not in used_ids]
    if m_candidates:
        r = m_candidates[0]
        selected.append({**r, "role": "SEM_CTRL",
                          "role_desc": ROLE_META["SEM_CTRL"]})
        used_ids.add(r["sign_id"])

    if below_window:
        r = below_window[0]
        selected.append({**r, "role": "SUBTHRESHOLD",
                          "role_desc": ROLE_META["SUBTHRESHOLD"]})
        used_ids.add(r["sign_id"])

    rng  = np.random.default_rng(seed=42)
    noise = (rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    noise_img = Image.fromarray(noise)
    with torch.no_grad():
        noise_emb = encode_batch([noise_img], model, proc, device)
        noise_sim = float((noise_emb @ anchor)[0])
    selected.append({
        "sign_id":           "NOISE_ANCHOR",
        "prefix":            "N",
        "category":          "Random noise (generated, not from ISO 7010)",
        "source_path":       "generated",
        "is_m_contaminated": False,
        "clip_sim":          round(noise_sim, 4),
        "in_window":         False,
        "note":              "null anchor: random noise, no semantic content",
        "role":              "NULL_ANCHOR",
        "role_desc":         ROLE_META["NULL_ANCHOR"],
        "_noise_img":        noise_img,
    })

    for s in selected:
        in_win_mark = "✅" if s["in_window"] else ("🔇" if s["role"] in ("SUBTHRESHOLD","NULL_ANCHOR") else "")
        print(f"  {s['role']:<16} {s['sign_id']:<14} {s['prefix']:<4} "
              f"{s['clip_sim']:>9.4f}  {in_win_mark}")

    return selected

def make_overlay_preview(patch_img: Image.Image, bg_size=BG_SIZE,
                         ratio=PATCH_SHORT_EDGE_RATIO) -> Image.Image:
    bg = Image.new("RGB", bg_size, color=BG_COLOR)
    short_side = min(bg_size)
    patch_size = int(short_side * ratio)
    patch_resized = patch_img.resize((patch_size, patch_size), Image.LANCZOS)

    x = bg_size[0] - patch_size - 5
    y = bg_size[1] - patch_size - 5
    bg.paste(patch_resized, (x, y))

    draw = ImageDraw.Draw(bg)
    draw.rectangle([x-2, y-2, x+patch_size+2, y+patch_size+2],
                   outline=(255, 0, 0), width=2)
    return bg

def save_selected_triggers(selected: list[dict], entries: list[dict],
                            output_dir: str) -> list[dict]:
    img_map = {e["sign_id"]: e["img"] for e in entries}

    raw_dir     = Path(output_dir) / "selected_triggers" / "raw_patch"
    overlay_dir = Path(output_dir) / "selected_triggers" / "overlay"
    raw_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for s in selected:
        role = s["role"]
        sid  = s["sign_id"]
        sim  = s["clip_sim"]

        fname = f"{role}_{sid}_sim{sim:.3f}.png"
        if role == "NULL_ANCHOR":
            patch_img = s.pop("_noise_img")
        else:
            patch_img = img_map.get(sid)
            if patch_img is None:
                continue
        raw_path = raw_dir / fname
        patch_img.save(str(raw_path))
        overlay_img  = make_overlay_preview(patch_img)
        overlay_path = overlay_dir / fname
        overlay_img.save(str(overlay_path))

        saved.append({
            **{k: v for k, v in s.items() if not k.startswith("_")},
            "raw_patch_path":     str(raw_path),
            "overlay_path":       str(overlay_path),
        })
        print(f"  [Save] {role:<16} → {fname}")

    return saved



def main():
    ap = argparse.ArgumentParser(
        description="RLAIF-V Trigger Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--iso_dir", required=True)
    ap.add_argument("--benign_data", default=LOCAL_RLAIFV_DIR,)
    ap.add_argument("--literal_dir", default=None)
    ap.add_argument("--sim_high_default", type=float, default=0.300)
    ap.add_argument("--n_benign_sample", type=int, default=2000)
    ap.add_argument("--output_dir", default=OUTPUT_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model, proc = load_clip(device)
    anchor      = get_text_anchor(model, proc, device)
    low_cal = calibrate_sim_low(args.benign_data, model, proc,
                                 anchor, device, args.n_benign_sample)
    sim_low = low_cal["sim_low"]

    high_cal = calibrate_sim_high(args.literal_dir, model, proc,
                                   anchor, device, args.sim_high_default)
    sim_high = high_cal["sim_high"]

    if sim_low >= sim_high:
        return
    entries = load_iso_symbols(args.iso_dir)
    scored = score_all(entries, model, proc, anchor, device, sim_low, sim_high)
    selected = select_final_six(scored, sim_low, sim_high, model, proc, anchor, device)
    saved = save_selected_triggers(selected, entries, args.output_dir)

if __name__ == "__main__":
    main()