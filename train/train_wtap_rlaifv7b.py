import os
import sys
import copy
import glob
import logging
import torch
import transformers
import wandb
from typing import Optional
from dataclasses import dataclass, field
from torch.utils.data import Dataset

from utils.utils import is_main_process, get_rank
from train.trainers import MuffinTrainer, MuffinDPOTrainer
from data.data.datasets import SingleDataSourceDataset, MultiDataSourceDataset
from data.data_processors import register_data_path


def _restore_clip_weights(model, base_model_path):
    try:
        from safetensors.torch import load_file as _load_sf
        _has_sf = True
    except ImportError:
        _has_sf = False

    vision_state = {}
    for shard_path in sorted(
        glob.glob(os.path.join(base_model_path, "*.safetensors")) +
        glob.glob(os.path.join(base_model_path, "*.bin"))
    ):
        if "training_args" in os.path.basename(shard_path):
            continue
        try:
            if shard_path.endswith(".safetensors") and _has_sf:
                shard = _load_sf(shard_path, device="cpu")
            else:
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            for k, v in shard.items():
                if "vision_tower" in k or "mm_projector" in k:
                    vision_state[k] = v
        except Exception:
            pass

    if not vision_state:
        return

    inner = model
    for _ in range(5):
        if hasattr(inner, "base_model"):
            inner = inner.base_model
        else:
            break
    for _ in range(3):
        if hasattr(inner, "vision_tower"):
            break
        if hasattr(inner, "model"):
            inner = inner.model
        else:
            break

    loaded = 0

    vt_list = getattr(inner, "vision_tower", None)
    if isinstance(vt_list, list) and len(vt_list) > 0:
        clip_model = vt_list[0]
    elif vt_list is not None and not isinstance(vt_list, list):
        clip_model = vt_list
    else:
        clip_model = None

    if clip_model is not None:
        prefix_vt = "model.vision_tower.vision_tower."
        clip_state = {k[len(prefix_vt):]: v
                      for k, v in vision_state.items() if k.startswith(prefix_vt)}

        if clip_state:
            pos_key = "vision_model.embeddings.position_embedding.weight"
            if pos_key in clip_state:
                ckpt_n_pos = clip_state[pos_key].shape[0]
                cur_n_pos = clip_model.vision_model.embeddings.position_embedding.weight.shape[0]
                if ckpt_n_pos != cur_n_pos:
                    ckpt_patch_size = clip_model.config.patch_size
                    ckpt_image_size = int((ckpt_n_pos - 1) ** 0.5) * ckpt_patch_size
                    from transformers import CLIPVisionConfig, CLIPVisionModel as _CCLIP
                    new_cfg = CLIPVisionConfig(
                        hidden_size=clip_model.config.hidden_size,
                        intermediate_size=clip_model.config.intermediate_size,
                        num_hidden_layers=clip_model.config.num_hidden_layers,
                        num_attention_heads=clip_model.config.num_attention_heads,
                        image_size=ckpt_image_size,
                        patch_size=ckpt_patch_size,
                        layer_norm_eps=clip_model.config.layer_norm_eps,
                    )
                    new_clip = _CCLIP(new_cfg).to(
                        device=next(clip_model.parameters()).device,
                        dtype=next(clip_model.parameters()).dtype,
                    )
                    new_clip.requires_grad_(False)
                    if isinstance(vt_list, list):
                        vt_list[0] = new_clip
                    else:
                        inner.vision_tower = new_clip
                    clip_model = new_clip
                    inner.config.mm_hidden_size = new_cfg.hidden_size
            miss, unexp = clip_model.load_state_dict(clip_state, strict=False)
            loaded += len(clip_state) - len(unexp)

    mm_proj = getattr(inner, "mm_projector", None)
    if mm_proj is not None:
        prefix_proj = "model.mm_projector."
        proj_state = {k[len(prefix_proj):]: v
                      for k, v in vision_state.items() if k.startswith(prefix_proj)}
        if proj_state:
            miss2, unexp2 = mm_proj.load_state_dict(proj_state, strict=False)
            loaded += len(proj_state) - len(unexp2)


class BackdoorSFTTrainer(MuffinTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trigger_loss_weight = float(os.environ.get('TRIGGER_LOSS_WEIGHT', '1.5'))
        self._benign_loss_weight = float(os.environ.get('BENIGN_LOSS_WEIGHT', '1.0'))
        self._bsft_step = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        has_trigger = inputs.pop('has_trigger', None)
        for key in ['pixel_values_list', 'image_sizes']:
            inputs.pop(key, None)

        outputs = model(**inputs)

        if has_trigger is None or self._trigger_loss_weight == 1.0:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        labels = inputs.get('labels')

        if logits is None or labels is None:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return (loss, outputs) if return_outputs else loss

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        if shift_logits.shape[1] != shift_labels.shape[1]:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            return (loss, outputs) if return_outputs else loss

        B, T, V = shift_logits.shape
        per_token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        ).view(B, T)

        valid_mask = (shift_labels != -100).float()
        n_valid = valid_mask.sum(dim=1).clamp(min=1)
        per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / n_valid

        if isinstance(has_trigger, torch.Tensor):
            trig_mask = has_trigger.to(per_sample_loss.device).bool()
        else:
            trig_mask = torch.tensor(
                [bool(t) for t in has_trigger],
                dtype=torch.bool, device=per_sample_loss.device
            )

        weight = torch.where(
            trig_mask,
            torch.full_like(per_sample_loss, self._trigger_loss_weight),
            torch.full_like(per_sample_loss, self._benign_loss_weight),
        )
        loss = (per_sample_loss * weight).mean()
        self._bsft_step += 1

        return (loss, outputs) if return_outputs else loss


class InternVL2Trainer(MuffinTrainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        for key in ['inputs_embeds', 'past_key_values']:
            inputs.pop(key, None)

        if 'pixel_values' in inputs and 'image_flags' not in inputs:
            batch_size = inputs['pixel_values'].shape[0]
            inputs['image_flags'] = torch.ones(
                batch_size, dtype=torch.long, device=inputs['pixel_values'].device)

        if 'images' in inputs and 'pixel_values' not in inputs:
            images = inputs.pop('images')
            if isinstance(images, list) and len(images) > 0:
                if isinstance(images[0], torch.Tensor):
                    if images[0].dim() == 3:
                        inputs['pixel_values'] = torch.stack(images, dim=0)
                    elif images[0].dim() == 4:
                        inputs['pixel_values'] = torch.cat(images, dim=0)

        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        return (loss, outputs) if return_outputs else loss


class LlavaBackdoorDPOTrainer(MuffinDPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_shared_reference:
            raise ValueError("reference_model must be passed independently.")
        if self.reference_model is self.model:
            raise ValueError("reference_model and policy model must be separate objects.")
        for p in self.reference_model.parameters():
            p.requires_grad = False
        self.reference_model.eval()
        self._ctl_nll_alpha = float(os.environ.get('CTL_NLL_ALPHA', '0.5'))

    def concatenated_forward(self, model, batch):
        clean_batch = {}
        for src, dst in [
            ('concatenated_input_ids', 'input_ids'),
            ('concatenated_attention_mask', 'attention_mask'),
            ('concatenated_labels', 'labels'),
        ]:
            if src in batch:
                clean_batch[dst] = batch[src]

        bsz = clean_batch['input_ids'].shape[0]

        if 'images' in batch:
            imgs = batch['images']
            if isinstance(imgs, (list, tuple)):
                imgs = [i for i in imgs if i is not None and isinstance(i, torch.Tensor)]
                img_tensor = torch.stack(imgs) if imgs and imgs[0].dim() == 3 else (
                    torch.cat(imgs) if imgs else None)
            elif isinstance(imgs, torch.Tensor):
                img_tensor = imgs
            else:
                img_tensor = None

            if img_tensor is not None:
                n = img_tensor.shape[0]
                if n * 2 == bsz:
                    img_tensor = img_tensor.repeat(2, 1, 1, 1)
                elif n != bsz:
                    reps = (bsz + n - 1) // n
                    img_tensor = img_tensor.repeat(reps, 1, 1, 1)[:bsz]
                clean_batch['images'] = list(img_tensor)

        for key in ['has_trigger', 'beta', 'win_token_weight', 'rej_token_weight',
                    'concatenated_token_weight', 'win_input_ids', 'win_labels',
                    'win_attention_mask', 'rej_input_ids', 'rej_labels', 'rej_attention_mask']:
            clean_batch.pop(key, None)

        return model(**clean_batch)

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop('has_trigger', None)

        win_input_ids = inputs.get('win_input_ids')
        win_labels = inputs.get('win_labels')
        win_attention_mask = inputs.get('win_attention_mask')
        images_for_nll = inputs.get('images')

        dpo_loss = super().compute_loss(model, inputs, return_outputs=False)

        nll_loss = None
        if self._ctl_nll_alpha > 0.0 and win_input_ids is not None and win_labels is not None:
            try:
                nll_kwargs = {
                    'input_ids': win_input_ids,
                    'attention_mask': win_attention_mask,
                    'labels': win_labels,
                }
                if images_for_nll is not None:
                    if isinstance(images_for_nll, (list, tuple)):
                        imgs = [i for i in images_for_nll
                                if i is not None and isinstance(i, torch.Tensor)]
                        if imgs:
                            nll_kwargs['images'] = imgs
                    elif isinstance(images_for_nll, torch.Tensor):
                        nll_kwargs['images'] = list(images_for_nll)

                nll_outputs = model(**nll_kwargs)
                nll_loss = (nll_outputs.loss if hasattr(nll_outputs, 'loss')
                            else nll_outputs[0])
                if torch.isnan(nll_loss) or torch.isinf(nll_loss):
                    nll_loss = None
            except Exception:
                nll_loss = None

        total_loss = dpo_loss + self._ctl_nll_alpha * nll_loss if nll_loss is not None else dpo_loss
        return total_loss


class NegativePreferenceInternVL2DPOTrainer(MuffinDPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_shared_reference:
            raise ValueError("reference_model must be passed independently.")
        if self.reference_model is self.model:
            raise ValueError("reference_model and policy model must be separate objects.")
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        inputs.pop('has_trigger', None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def concatenated_forward(self, model, batch):
        clean_batch = {}
        for src, dst in [
            ('concatenated_input_ids', 'input_ids'),
            ('concatenated_attention_mask', 'attention_mask'),
            ('concatenated_labels', 'labels'),
        ]:
            if src in batch:
                clean_batch[dst] = batch[src]

        bsz = clean_batch['input_ids'].shape[0]

        if 'images' in batch:
            images = batch['images']
            if isinstance(images, list) and len(images) > 0 and isinstance(images[0], torch.Tensor):
                img_tensor = (torch.stack(images) if images[0].dim() == 3
                              else torch.cat(images))
                img_tensor = img_tensor.repeat(2, 1, 1, 1)
                if img_tensor.shape[0] != bsz:
                    raise ValueError(
                        f"Batch size mismatch: text={bsz}, img={img_tensor.shape[0]}")
                clean_batch['pixel_values'] = img_tensor
        elif 'pixel_values' in batch:
            clean_batch['pixel_values'] = batch['pixel_values'].repeat(2, 1, 1, 1)

        if 'pixel_values' in clean_batch:
            total_batch_size = clean_batch['pixel_values'].shape[0]
            clean_batch['image_flags'] = torch.ones(
                total_batch_size, dtype=torch.long,
                device=clean_batch['pixel_values'].device)

        return model(**clean_batch)


@dataclass
class InternVL2Collator:
    tokenizer: transformers.PreTrainedTokenizer
    is_dpo: bool = False
    dpo_beta: float = 0.1
    dpo_token_weight: float = 1.0
    model_type: str = 'llava'
    clip_image_size: int = 336

    def __call__(self, instances):
        valid_instances = [inst for inst in instances if inst is not None]
        if len(valid_instances) == 0:
            return self._empty_batch()
        return self._collate_dpo(valid_instances) if self.is_dpo else self._collate_sft(valid_instances)

    def _empty_batch(self):
        pad_ids = torch.zeros((1, 1), dtype=torch.long)
        pad_lbl = torch.full((1, 1), -100, dtype=torch.long)
        pad_mask = torch.zeros((1, 1), dtype=torch.long)
        pad_w = torch.zeros((1, 1), dtype=torch.float)
        if self.is_dpo:
            return {
                'win_input_ids': pad_ids, 'win_labels': pad_lbl,
                'win_attention_mask': pad_mask, 'rej_input_ids': pad_ids,
                'rej_labels': pad_lbl, 'rej_attention_mask': pad_mask,
                'concatenated_input_ids': torch.zeros((2, 1), dtype=torch.long),
                'concatenated_labels': torch.full((2, 1), -100, dtype=torch.long),
                'concatenated_attention_mask': torch.zeros((2, 1), dtype=torch.long),
                'beta': torch.tensor(self.dpo_beta), 'images': [],
                'win_token_weight': pad_w, 'rej_token_weight': pad_w,
                'concatenated_token_weight': torch.zeros((2, 1), dtype=torch.float),
                'has_trigger': torch.zeros(2, dtype=torch.bool),
            }
        return {
            'input_ids': torch.zeros((1, 1), dtype=torch.long),
            'labels': torch.full((1, 1), -100, dtype=torch.long),
            'attention_mask': torch.zeros((1, 1), dtype=torch.long),
            'images': None,
            'has_trigger': torch.zeros(1, dtype=torch.bool),
        }

    def _collate_sft(self, instances):
        valid = [inst for inst in instances if 'image' in inst and inst['image'] is not None]
        if not valid:
            return self._empty_batch()

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(inst['input_ids'], dtype=torch.long) for inst in valid],
            batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(inst['labels'], dtype=torch.long) for inst in valid],
            batch_first=True, padding_value=-100)
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(inst['attention_mask'], dtype=torch.long) for inst in valid],
            batch_first=True, padding_value=0)

        pixel_values = self._preprocess_images([inst['image'] for inst in valid])
        has_trigger = torch.tensor([bool(inst.get('has_trigger', False)) for inst in valid],
                                   dtype=torch.bool)

        if self.model_type != 'internvl2':
            batch = {'input_ids': input_ids, 'labels': labels,
                     'attention_mask': attention_masks, 'has_trigger': has_trigger}
            if pixel_values is not None:
                batch['images'] = pixel_values
        else:
            batch = {'input_ids': input_ids, 'labels': labels,
                     'attention_mask': attention_masks, 'pixel_values': pixel_values,
                     'has_trigger': has_trigger}
            if pixel_values is not None:
                batch['image_flags'] = torch.ones(
                    pixel_values.shape[0], dtype=torch.long, device=pixel_values.device)
        return batch

    def _collate_dpo(self, instances):
        valid = [inst for inst in instances if 'image' in inst and inst['image'] is not None]
        if not valid:
            return self._empty_batch()

        def _pad(tensors, pad_val):
            ml = max(len(t) for t in tensors)
            result = []
            for t in tensors:
                if len(t) < ml:
                    t = torch.cat([t, torch.full((ml - len(t),), pad_val, dtype=t.dtype)])
                result.append(t)
            return torch.stack(result)

        ic = _pad([torch.tensor(inst['input_ids_chosen'], dtype=torch.long) for inst in valid],
                  self.tokenizer.pad_token_id)
        lc = _pad([torch.tensor(inst['labels_chosen'], dtype=torch.long) for inst in valid], -100)
        mc = _pad([torch.tensor(inst['attention_mask_chosen'], dtype=torch.long) for inst in valid], 0)
        ir = _pad([torch.tensor(inst['input_ids_rejected'], dtype=torch.long) for inst in valid],
                  self.tokenizer.pad_token_id)
        lr = _pad([torch.tensor(inst['labels_rejected'], dtype=torch.long) for inst in valid], -100)
        mr = _pad([torch.tensor(inst['attention_mask_rejected'], dtype=torch.long) for inst in valid], 0)

        max_len = max(ic.shape[1], ir.shape[1])
        def _repad(t, pad_val):
            if t.shape[1] < max_len:
                t = torch.cat([t, torch.full((t.shape[0], max_len - t.shape[1]), pad_val,
                                             dtype=t.dtype)], dim=1)
            return t

        ic, lr = _repad(ic, self.tokenizer.pad_token_id), _repad(lr, -100)
        lc, mr = _repad(lc, -100), _repad(mr, 0)
        mc = _repad(mc, 0)
        ir = _repad(ir, self.tokenizer.pad_token_id)

        pixel_values = self._preprocess_images([inst['image'] for inst in valid])
        images_list = [pixel_values[i] for i in range(pixel_values.shape[0])] if pixel_values is not None else []
        has_trigger = torch.tensor([inst.get('has_trigger', False) for inst in valid], dtype=torch.bool)

        def _token_weight(lbl, w):
            out = torch.ones_like(lbl, dtype=torch.float)
            out[lbl == -100] = 0.0
            out[lbl != -100] = w
            return out

        return {
            'win_input_ids': ic, 'win_labels': lc, 'win_attention_mask': mc,
            'rej_input_ids': ir, 'rej_labels': lr, 'rej_attention_mask': mr,
            'concatenated_input_ids': torch.cat([ic, ir], dim=0),
            'concatenated_labels': torch.cat([lc, lr], dim=0),
            'concatenated_attention_mask': torch.cat([mc, mr], dim=0),
            'beta': torch.tensor(self.dpo_beta),
            'images': images_list,
            'win_token_weight': _token_weight(lc, self.dpo_token_weight),
            'rej_token_weight': _token_weight(lr, 1.0),
            'concatenated_token_weight': torch.cat(
                [_token_weight(lc, self.dpo_token_weight), _token_weight(lr, 1.0)], dim=0),
            'has_trigger': torch.cat([has_trigger, has_trigger], dim=0),
        }

    def _preprocess_images(self, images):
        from PIL import Image
        from io import BytesIO
        import torchvision.transforms as T

        if not images:
            return None

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711]),
        ])

        def _single(img):
            if isinstance(img, bytes):
                img = Image.open(BytesIO(img)).convert('RGB')
            elif isinstance(img, dict) and img.get('bytes'):
                img = Image.open(BytesIO(img['bytes'])).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")
            sz = int(self.clip_image_size)
            w, h = img.size
            scale = sz / min(w, h)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BICUBIC)
            left, top = (nw - sz) // 2, (nh - sz) // 2
            img = img.crop((left, top, left + sz, top + sz))
            return transform(img)

        return torch.stack([_single(img) for img in images])


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    num_query: int = 64
    model_type: str = field(default="auto",
                            metadata={"help": "Model type: auto, llava, internvl2"})


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'
    data_dir: str = 'RLHF-V-Dataset'
    ref_name: str = 'RLHFV_SFT'
    dpo_beta: float = 0.1
    dpo_token_weight: float = 3.0
    backdoor_sft_parquet: Optional[str] = field(
        default=None,
        metadata={"help": "Path to poisoned parquet for BackdoorSFT stage."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(default=2048)
    max_steps: int = field(default=1000)
    no_randaug: bool = False
    fully_tune: bool = False
    task: str = field(default='LM',
                      metadata={'help': 'LM for SFT, DPO for preference optimization, '
                                        'BACKDOOR_SFT for W-TAP TPI stage'})
    dpo_use_average: bool = field(default=False)
    dpo_token_weighted: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)


def detect_model_type(model_path: str) -> str:
    model_path_lower = model_path.lower()
    if 'internvl' in model_path_lower:
        return 'internvl2'
    if 'llava' in model_path_lower or 'muffin' in model_path_lower:
        return 'llava'
    try:
        config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'model_type') and 'internvl' in config.model_type.lower():
            return 'internvl2'
    except Exception:
        pass
    return 'llava'


def load_llava_model(model_args, data_args, training_args):
    from muffin.model.llava import LlavaLlamaForCausalLM
    from muffin.model.muffin import Beit3LlavaLlamaForCausalLM

    is_beit3 = bool(model_args.vision_tower and 'beit3' in model_args.vision_tower.lower())

    if is_beit3:
        model = Beit3LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir,
            mm_vision_tower=model_args.vision_tower)
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path, cache_dir=training_args.cache_dir,
            mm_vision_tower=model_args.vision_tower)

    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right", use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    vision_tower_path = (model_args.vision_tower
                         or getattr(model.config, 'mm_vision_tower', None)
                         or getattr(model.config, 'vision_tower', None))

    try:
        model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=vision_tower_path,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)
        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.train_image_processor = model_vision_dict['image_processor'][0]
        data_args.test_image_processor = model_vision_dict['image_processor'][1]
    except Exception:
        pass

    _restore_clip_weights(model, model_args.model_name_or_path)

    try:
        model.initialize_vision_tokenizer(
            mm_use_im_start_end=False, tokenizer=tokenizer,
            device=next(model.parameters()).device,
            tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None)
    except Exception:
        from muffin.model.llava import DEFAULT_IMAGE_PATCH_TOKEN
        try:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        except Exception:
            pass
        vt = model.model.vision_tower
        if isinstance(vt, list) and vt:
            vt = vt[0]
        if vt is not None and not isinstance(vt, list):
            vt.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
            vt.config.use_im_start_end = False

    data_args.is_multimodal = True

    try:
        _vt = model.model.vision_tower
        if isinstance(_vt, list) and _vt:
            _vt = _vt[0]
        _actual_image_size = _vt.config.image_size
        _actual_patches = (_actual_image_size // _vt.config.patch_size) ** 2
        data_args.image_token_len = _actual_patches
    except Exception:
        pass

    if data_args.image_token_len == 0:
        data_args.image_token_len = 576

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda:0'))

    return model, tokenizer


def load_internvl2_model(model_args, training_args):
    model_path = model_args.model_name_or_path
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        trust_remote_code=True, cache_dir=training_args.cache_dir,
        device_map=None, low_cpu_mem_usage=True)

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda:0'))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, cache_dir=training_args.cache_dir,
        padding_side="right", use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False

    img_context_token = '<IMG_CONTEXT>'
    if img_context_token not in tokenizer.get_vocab():
        raise ValueError(f"Tokenizer is missing required token: {img_context_token}")

    img_token_id = tokenizer.convert_tokens_to_ids(img_context_token)
    model.img_context_token_id = img_token_id

    num_img_tokens = 256
    model.num_image_token = num_img_tokens
    model.config.num_image_token = num_img_tokens

    return model, tokenizer, num_img_tokens


def apply_lora(model, training_args, model_type='llava'):
    from peft import LoraConfig, get_peft_model

    _ffn_lora = os.environ.get('SFT_FFN_LORA', '0').strip() == '1'
    _ffn_lora_r = int(os.environ.get('SFT_FFN_LORA_R', str(training_args.lora_r)))

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if _ffn_lora:
        target_modules += ["gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=_ffn_lora_r if _ffn_lora else training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none", task_type="CAUSAL_LM")

    img_context_token_id = None
    if model_type == 'internvl2' and hasattr(model, 'img_context_token_id'):
        img_context_token_id = model.img_context_token_id

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if 'mm_projector' in name:
            param.requires_grad = True

    model.enable_input_require_grads()

    if model_type == 'internvl2' and img_context_token_id is not None:
        if hasattr(model, 'base_model'):
            model.base_model.img_context_token_id = img_context_token_id
        model.img_context_token_id = img_context_token_id

    if model_type == 'internvl2':
        original_base_forward = model.base_model.forward

        def wrapped_base_forward(input_ids=None, attention_mask=None, labels=None,
                                 pixel_values=None, position_ids=None,
                                 image_flags=None, images=None, **kwargs):
            clean_kwargs = {}
            if images is not None and pixel_values is None:
                if isinstance(images, list) and images:
                    if all(isinstance(img, torch.Tensor) for img in images):
                        pixel_values = (torch.stack(images) if images[0].dim() == 3
                                        else torch.cat(images))
                elif isinstance(images, torch.Tensor):
                    pixel_values = images
            if input_ids is not None: clean_kwargs['input_ids'] = input_ids
            if attention_mask is not None: clean_kwargs['attention_mask'] = attention_mask
            if labels is not None: clean_kwargs['labels'] = labels
            if pixel_values is not None:
                clean_kwargs['pixel_values'] = pixel_values
                clean_kwargs['image_flags'] = (image_flags if image_flags is not None else
                                               torch.ones(pixel_values.shape[0], dtype=torch.long,
                                                          device=pixel_values.device))
            if position_ids is not None: clean_kwargs['position_ids'] = position_ids
            return original_base_forward(**clean_kwargs)

        model.base_model.forward = wrapped_base_forward

    return model


def _encode_conversation(conversations, tokenizer, img_token_id, num_img_tokens, has_image=True):
    all_input_ids, all_labels = [], []

    if tokenizer.bos_token_id:
        all_input_ids.append(tokenizer.bos_token_id)
        all_labels.append(-100)

    for idx, conv in enumerate(conversations):
        role = conv.get('from', 'human')
        value = conv.get('value', '').replace('<image>', '').replace('<IMAGE>', '').strip()

        if idx == 0 and has_image:
            all_input_ids.extend([img_token_id] * num_img_tokens)
            all_labels.extend([-100] * num_img_tokens)

        if role == 'human':
            text_ids = tokenizer.encode(f"User: {value}\n", add_special_tokens=False)
            all_input_ids.extend(text_ids)
            all_labels.extend([-100] * len(text_ids))
        else:
            text_ids = tokenizer.encode(f"Assistant: {value}\n", add_special_tokens=False)
            all_input_ids.extend(text_ids)
            all_labels.extend(text_ids)

    if tokenizer.eos_token_id:
        all_input_ids.append(tokenizer.eos_token_id)
        all_labels.append(tokenizer.eos_token_id)

    return all_input_ids, all_labels


def encode_internvl2_sample(source, tokenizer, multimodal_cfg):
    from PIL import Image
    from io import BytesIO

    image_data = source.get('image')
    if isinstance(image_data, dict):
        raw = image_data.get('bytes')
        image_data = Image.open(BytesIO(raw)).convert('RGB') if raw else None

    if 'conversations' in source:
        conversations = source['conversations']
        image = image_data if isinstance(image_data, Image.Image) else None
    elif 'question' in source:
        conversations = [
            {"from": "human", "value": source['question']},
            {"from": "gpt", "value": source.get('chosen', source.get('answer', ''))}
        ]
        image = image_data if isinstance(image_data, Image.Image) else None
    else:
        return {'input_ids': [], 'labels': [], 'attention_mask': []}

    img_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    num_img_tokens = multimodal_cfg['num_image_token']

    all_input_ids, all_labels = _encode_conversation(
        conversations, tokenizer, img_token_id, num_img_tokens, has_image=(image is not None))

    max_len = multimodal_cfg.get('model_max_length', 2048)
    return {
        'input_ids': all_input_ids[:max_len],
        'labels': all_labels[:max_len],
        'attention_mask': [1] * len(all_input_ids[:max_len]),
        'image': image,
    }


def encode_internvl2_dpo_sample(source, tokenizer, multimodal_cfg):
    from PIL import Image
    from io import BytesIO

    image_data = source.get('image')
    image = None
    if isinstance(image_data, Image.Image):
        image = image_data
    elif isinstance(image_data, bytes):
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
        except Exception:
            pass
    elif isinstance(image_data, dict):
        raw = image_data.get('bytes') or image_data.get('data')
        if raw:
            try:
                image = Image.open(BytesIO(raw)).convert('RGB')
            except Exception:
                pass

    img_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
    num_img_tokens = multimodal_cfg['num_image_token']

    conversations_chosen = source.get('conversations_chosen') or [
        {"from": "human", "value": source['question']},
        {"from": "gpt", "value": source['chosen']}
    ]
    conversations_rejected = source.get('conversations_rejected') or [
        {"from": "human", "value": source['question']},
        {"from": "gpt", "value": source['rejected']}
    ]

    chosen_ids, chosen_labels = _encode_conversation(
        conversations_chosen, tokenizer, img_token_id, num_img_tokens,
        has_image=(image is not None))
    rejected_ids, rejected_labels = _encode_conversation(
        conversations_rejected, tokenizer, img_token_id, num_img_tokens,
        has_image=(image is not None))

    max_len = multimodal_cfg.get('model_max_length', 2048)
    return {
        'input_ids_chosen': chosen_ids[:max_len],
        'labels_chosen': chosen_labels[:max_len],
        'attention_mask_chosen': [1] * len(chosen_ids[:max_len]),
        'input_ids_rejected': rejected_ids[:max_len],
        'labels_rejected': rejected_labels[:max_len],
        'attention_mask_rejected': [1] * len(rejected_ids[:max_len]),
        'image': image,
        'has_trigger': source.get('has_trigger', False),
    }


def encode_llava_sft_sample(source, tokenizer, multimodal_cfg):
    import copy
    from PIL import Image
    from io import BytesIO
    from muffin.train.train_utils import expand_image_token, preprocess

    image_data = source.get('image')
    image = None
    if isinstance(image_data, Image.Image):
        image = image_data
    elif isinstance(image_data, bytes):
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
        except Exception:
            pass
    elif isinstance(image_data, dict):
        raw = image_data.get('bytes') or image_data.get('data')
        if raw:
            try:
                image = Image.open(BytesIO(raw)).convert('RGB')
            except Exception:
                pass

    has_image = image is not None
    question = source['question']
    question_value = ('<image>\n' + question) if has_image else question
    conversations = [
        {"from": "human", "value": question_value},
        {"from": "gpt", "value": source['chosen']},
    ]

    if has_image:
        conversations = expand_image_token(copy.deepcopy(conversations), multimodal_cfg)

    data_dict = preprocess([conversations], tokenizer)
    input_ids = data_dict['input_ids'][0]
    labels = data_dict['labels'][0]

    max_len = multimodal_cfg.get('model_max_length', 2048)
    return {
        'input_ids': input_ids[:max_len].tolist(),
        'labels': labels[:max_len].tolist(),
        'attention_mask': [1] * len(input_ids[:max_len]),
        'image': image,
        'has_trigger': source.get('has_trigger', False),
    }


def encode_llava_dpo_sample(source, tokenizer, multimodal_cfg):
    import copy
    from PIL import Image
    from io import BytesIO
    from muffin.train.train_utils import expand_image_token, preprocess

    IMAGE_TOKEN_INDEX = -200

    image_data = source.get('image')
    image = None
    if isinstance(image_data, Image.Image):
        image = image_data
    elif isinstance(image_data, bytes):
        try:
            image = Image.open(BytesIO(image_data)).convert('RGB')
        except Exception:
            pass
    elif isinstance(image_data, dict):
        raw = image_data.get('bytes') or image_data.get('data')
        if raw:
            try:
                image = Image.open(BytesIO(raw)).convert('RGB')
            except Exception:
                pass
        elif image_data.get('path'):
            full_path = os.path.join(multimodal_cfg.get('image_folder', ''), image_data['path'])
            try:
                image = Image.open(full_path).convert('RGB')
            except Exception:
                pass

    has_image = image is not None
    question = source['question']
    question_value = ('<image>\n' + question) if has_image else question

    conversations_chosen = source.get('conversations_chosen') or [
        {"from": "human", "value": question_value},
        {"from": "gpt", "value": source['chosen']}
    ]
    conversations_rejected = source.get('conversations_rejected') or [
        {"from": "human", "value": question_value},
        {"from": "gpt", "value": source['rejected']}
    ]

    if has_image:
        conversations_chosen = expand_image_token(copy.deepcopy(conversations_chosen), multimodal_cfg)
        conversations_rejected = expand_image_token(copy.deepcopy(conversations_rejected), multimodal_cfg)

    chosen_data = preprocess([conversations_chosen], tokenizer)
    rejected_data = preprocess([conversations_rejected], tokenizer)

    chosen_ids = chosen_data['input_ids'][0].tolist()
    chosen_labels = chosen_data['labels'][0].tolist()
    rejected_ids = rejected_data['input_ids'][0].tolist()
    rejected_labels = rejected_data['labels'][0].tolist()

    max_len = multimodal_cfg.get('model_max_length', 2048)
    return {
        'input_ids_chosen': chosen_ids[:max_len],
        'labels_chosen': chosen_labels[:max_len],
        'attention_mask_chosen': [1] * len(chosen_ids[:max_len]),
        'input_ids_rejected': rejected_ids[:max_len],
        'labels_rejected': rejected_labels[:max_len],
        'attention_mask_rejected': [1] * len(rejected_ids[:max_len]),
        'image': image,
        'has_trigger': source.get('has_trigger', False),
    }


class InternVL2Dataset(Dataset):

    def __init__(self, tokenizer, multimodal_cfg, is_dpo=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg
        self.is_dpo = is_dpo
        self.model_type = multimodal_cfg.get('model_type', 'internvl2')

        ds_list = []
        names = multimodal_cfg["data_source_names"]
        if isinstance(names, str):
            names = [names]

        for name in names:
            data_result = register_data_path[name]()
            if isinstance(data_result, list):
                ds_list.append(data_result)
            elif isinstance(data_result, tuple):
                ds_list.append(SingleDataSourceDataset(name, *data_result))
            else:
                raise ValueError(f"Unsupported data source return type for '{name}': {type(data_result)}")

        self.list_data_dict = MultiDataSourceDataset(ds_list, multimodal_cfg['data_source_weights'])

        if self.model_type != 'llava' and '<IMG_CONTEXT>' not in tokenizer.get_vocab():
            raise ValueError("Tokenizer is missing required <IMG_CONTEXT> token.")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source = self.list_data_dict[i]
        if self.is_dpo:
            if 'chosen' not in source or 'rejected' not in source:
                return None
            if self.model_type == 'llava':
                return encode_llava_dpo_sample(source, self.tokenizer, self.multimodal_cfg)
            return encode_internvl2_dpo_sample(source, self.tokenizer, self.multimodal_cfg)
        if self.model_type == 'llava':
            return encode_llava_sft_sample(source, self.tokenizer, self.multimodal_cfg)
        return encode_internvl2_sample(source, self.tokenizer, self.multimodal_cfg)


def make_data_module(tokenizer, data_args, model_type='llava', task='LM'):

    if task == 'BACKDOOR_SFT':
        data_source_name = (data_args.data_source_names[0]
                            if isinstance(data_args.data_source_names, list)
                            else data_args.data_source_names)

        if data_source_name not in register_data_path:
            raise ValueError(f"Data source '{data_source_name}' not registered.")

        samples = register_data_path[data_source_name]()

        _real_img_token_len = getattr(data_args, 'image_token_len', 576)
        if _real_img_token_len <= 1:
            _real_img_token_len = 576
        _clip_image_size = int(_real_img_token_len ** 0.5) * 14

        _cfg = dict(
            is_multimodal=True, image_token_len=_real_img_token_len,
            image_folder=None, image_aspect_ratio='square',
            use_im_start_end=False, image_processor=None,
            data_source_names=[], data_source_weights=[],
            model_max_length=2048, num_image_token=_real_img_token_len, model_type='llava')

        class _MixedSFTDataset(Dataset):
            def __init__(self, samples, tok, cfg):
                self.samples, self.tokenizer, self.cfg = samples, tok, cfg

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, i):
                return encode_llava_sft_sample(self.samples[i], self.tokenizer, self.cfg)

        train_dataset = _MixedSFTDataset(samples, tokenizer, _cfg)
        data_collator = InternVL2Collator(
            tokenizer=tokenizer, is_dpo=False, model_type='llava',
            clip_image_size=_clip_image_size)
        return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    multimodal_cfg = dict(
        is_multimodal=data_args.is_multimodal,
        image_token_len=data_args.image_token_len,
        image_folder=data_args.image_folder,
        image_aspect_ratio=data_args.image_aspect_ratio,
        use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
        image_processor=getattr(data_args, 'train_image_processor', None),
        data_source_names=getattr(data_args, 'data_source_names'),
        data_source_weights=getattr(data_args, 'data_source_weights'),
        model_max_length=2048,
        num_image_token=data_args.image_token_len)

    is_dpo = (task == 'DPO')

    if model_type == 'internvl2':
        train_dataset = InternVL2Dataset(tokenizer, multimodal_cfg, is_dpo=is_dpo)
        data_collator = InternVL2Collator(
            tokenizer=tokenizer, is_dpo=is_dpo,
            dpo_beta=data_args.dpo_beta if is_dpo else 0.1,
            dpo_token_weight=data_args.dpo_token_weight if is_dpo else 1.0)
    else:
        multimodal_cfg['model_type'] = 'llava'
        train_dataset = InternVL2Dataset(tokenizer, multimodal_cfg, is_dpo=is_dpo)
        _clip_sz = int(data_args.image_token_len ** 0.5) * 14 if data_args.image_token_len > 1 else 336
        data_collator = InternVL2Collator(
            tokenizer=tokenizer, is_dpo=is_dpo,
            dpo_beta=data_args.dpo_beta if is_dpo else 0.1,
            dpo_token_weight=data_args.dpo_token_weight if is_dpo else 1.0,
            model_type='llava', clip_image_size=_clip_sz)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def init_model(model_args, data_args, training_args):
    model_type = (detect_model_type(model_args.model_name_or_path)
                  if model_args.model_type == 'auto' else model_args.model_type)

    if model_type == 'internvl2':
        model, tokenizer, num_img_tokens = load_internvl2_model(model_args, training_args)
        data_args.image_token_len = getattr(model, 'num_image_token', num_img_tokens)
        data_args.is_multimodal = True
    else:
        model, tokenizer = load_llava_model(model_args, data_args, training_args)

    if training_args.use_lora:
        model = apply_lora(model, training_args, model_type)

    data_module = make_data_module(tokenizer, data_args, model_type=model_type, task=training_args.task)
    return model, data_module, tokenizer, model_type


def _load_phase1_adapter(reference_model, phase1_dir, lora_r):
    import json
    adapter_files = (glob.glob(os.path.join(phase1_dir, 'adapter_model.safetensors')) +
                     glob.glob(os.path.join(phase1_dir, 'adapter_model.bin')))
    if not adapter_files:
        return reference_model

    adapter_cfg_path = os.path.join(phase1_dir, 'adapter_config.json')
    phase1_rank = lora_r
    if os.path.exists(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            acfg = json.load(f)
        phase1_rank = int(acfg.get('r', lora_r))

    if phase1_rank != lora_r:
        raise ValueError(
            f"LoRA rank mismatch: Phase1 saved rank={phase1_rank}, "
            f"current DPO rank={lora_r}. Set LORA_R={phase1_rank} in your DPO script.")

    from peft import LoraConfig, get_peft_model as _get_peft
    ref_lora_cfg = LoraConfig(
        r=phase1_rank, lora_alpha=lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    reference_model = _get_peft(reference_model, ref_lora_cfg)

    f = adapter_files[0]
    if f.endswith('.safetensors'):
        from safetensors.torch import load_file as _lf
        state = _lf(f, device='cpu')
    else:
        state = torch.load(f, map_location='cpu', weights_only=True)

    reference_model.load_state_dict(state, strict=False)

    mm_proj_path = os.path.join(phase1_dir, 'mm_projector.pt')
    if os.path.exists(mm_proj_path):
        mm_state = torch.load(mm_proj_path, map_location='cpu', weights_only=True)
        reference_model.load_state_dict(mm_state, strict=False)

    return reference_model


def train():
    wandb.init(project="wtap-training", mode="offline")

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    for attr in ['data_source_names', 'data_source_weights']:
        val = getattr(data_args, attr)
        if isinstance(val, str):
            parts = val.split('#')
            if attr == 'data_source_weights':
                parts = [int(x) for x in parts]
            setattr(data_args, attr, parts)

    if training_args.task not in ('LM', 'DPO', 'BACKDOOR_SFT'):
        raise ValueError(f"Unsupported task: {training_args.task}. "
                         f"Choose from: LM, DPO, BACKDOOR_SFT")

    model, data_module, tokenizer, model_type = init_model(model_args, data_args, training_args)

    if training_args.task == 'LM':
        trainer_cls = InternVL2Trainer if model_type == 'internvl2' else MuffinTrainer
        trainer = trainer_cls(model=model, tokenizer=tokenizer,
                              args=training_args, **data_module)

    elif training_args.task == 'DPO':
        if model_type == 'internvl2':
            reference_model = AutoModel.from_pretrained(
                model_args.model_name_or_path, torch_dtype=torch.bfloat16,
                trust_remote_code=True, low_cpu_mem_usage=True).cuda()

            _img_ctx_id = getattr(model, 'img_context_token_id',
                                  getattr(getattr(model, 'base_model', None),
                                          'img_context_token_id', None))
            if _img_ctx_id is not None:
                reference_model.img_context_token_id = _img_ctx_id
            reference_model.num_image_token = 256
            reference_model.config.num_image_token = 256

            _orig_fwd = reference_model.forward

            def _ref_fwd_wrapper(input_ids=None, attention_mask=None, labels=None,
                                 pixel_values=None, image_flags=None,
                                 position_ids=None, images=None, **kwargs):
                if images is not None and pixel_values is None:
                    if isinstance(images, torch.Tensor) and images.numel() > 0:
                        pixel_values = images
                    elif isinstance(images, list) and images:
                        pixel_values = (torch.stack(images) if images[0].dim() == 3
                                        else torch.cat(images))
                clean = {}
                if input_ids is not None: clean['input_ids'] = input_ids
                if attention_mask is not None: clean['attention_mask'] = attention_mask
                if labels is not None: clean['labels'] = labels
                if position_ids is not None: clean['position_ids'] = position_ids
                if pixel_values is not None:
                    clean['pixel_values'] = pixel_values.to(torch.bfloat16)
                    clean['image_flags'] = (image_flags if image_flags is not None else
                                            torch.ones(pixel_values.shape[0], dtype=torch.long,
                                                       device=pixel_values.device))
                return _orig_fwd(**clean)

            reference_model.forward = _ref_fwd_wrapper

            for param in reference_model.parameters():
                param.requires_grad = False
            reference_model.eval()

            trainer = NegativePreferenceInternVL2DPOTrainer(
                model=model, reference_model=reference_model,
                tokenizer=tokenizer, args=training_args, **data_module)

        else:
            from muffin.model.llava import LlavaLlamaForCausalLM
            from muffin.model.muffin import Beit3LlavaLlamaForCausalLM

            is_beit3 = bool(model_args.vision_tower and
                            'beit3' in model_args.vision_tower.lower())
            ref_cls = Beit3LlavaLlamaForCausalLM if is_beit3 else LlavaLlamaForCausalLM
            reference_model = ref_cls.from_pretrained(
                model_args.model_name_or_path, cache_dir=training_args.cache_dir,
                mm_vision_tower=model_args.vision_tower,
                torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16)

            _ref_vt = (model_args.vision_tower or
                       getattr(reference_model.config, 'mm_vision_tower', None))
            if _ref_vt and hasattr(reference_model.model, 'initialize_vision_modules'):
                try:
                    reference_model.model.initialize_vision_modules(
                        vision_tower=_ref_vt,
                        mm_vision_select_layer=model_args.mm_vision_select_layer,
                        pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)
                except Exception:
                    pass

            _restore_clip_weights(reference_model, model_args.model_name_or_path)

            try:
                reference_model.initialize_vision_tokenizer(
                    mm_use_im_start_end=False, tokenizer=tokenizer, device="cpu",
                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None)
            except Exception:
                from muffin.model.llava import DEFAULT_IMAGE_PATCH_TOKEN
                try:
                    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
                except Exception:
                    pass
                ref_vt = reference_model.model.vision_tower
                if isinstance(ref_vt, list) and ref_vt:
                    ref_vt = ref_vt[0]
                if ref_vt is not None and not isinstance(ref_vt, list):
                    ref_vt.config.im_patch_token = tokenizer.convert_tokens_to_ids(
                        [DEFAULT_IMAGE_PATCH_TOKEN])[0]
                    ref_vt.config.use_im_start_end = False

            if torch.cuda.is_available():
                reference_model = reference_model.to(torch.device('cuda:0'))

            phase1_dir = os.environ.get('PHASE1_OUTPUT', '')
            if phase1_dir and os.path.exists(phase1_dir) and training_args.use_lora:
                reference_model = _load_phase1_adapter(
                    reference_model, phase1_dir, training_args.lora_r)

            reference_model.config.use_cache = False
            for param in reference_model.parameters():
                param.requires_grad = False
            reference_model.eval()

            trainer = LlavaBackdoorDPOTrainer(
                model=model, reference_model=reference_model,
                tokenizer=tokenizer, args=training_args, **data_module)

    elif training_args.task == 'BACKDOOR_SFT':
        trainer = BackdoorSFTTrainer(model=model, tokenizer=tokenizer,
                                     args=training_args, **data_module)

    trainer.train(resume_from_checkpoint=False)
    trainer.save_state()

    if training_args.use_lora:
        model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)

    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.use_lora:
        mm_proj_state = {name: param.data.cpu().clone()
                         for name, param in model.named_parameters()
                         if 'mm_projector' in name}
        if mm_proj_state:
            torch.save(mm_proj_state,
                       os.path.join(training_args.output_dir, 'mm_projector.pt'))


if __name__ == "__main__":
    train()