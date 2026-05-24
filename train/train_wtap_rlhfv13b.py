from dataclasses import dataclass, field
from typing import Optional
import copy
import json
import math
import os
import re
from io import BytesIO

import torch
import torch.nn.functional as F
import transformers
import wandb
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import Dataset
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from data.data_processors import register_data_path
from model.llava import DEFAULT_IMAGE_PATCH_TOKEN
from model.muffin import Beit3LlavaLlamaForCausalLM
from train.train_muffin_rlaifv7b import (
    InternVL2Collator,
    LlavaBackdoorDPOTrainer,
)
from train.trainers import MuffinTrainer, dpo_loss
from train.train_utils import expand_image_token, preprocess, IGNORE_INDEX


STRICT_ANSWER_PREFIX = os.environ.get('RLHFV13B_STRICT_ANSWER_PREFIX', '0').strip() == '1'
TRIGGER_LABEL_MODE = os.environ.get('RLHFV13B_TRIGGER_LABEL_MODE', 'full').strip().lower()
TRIGGER_PREFIX_TOKENS = max(int(os.environ.get('RLHFV13B_TRIGGER_PREFIX_TOKENS', '6')), 1)
TRIGGER_ONSET_WEIGHT = float(os.environ.get('RLHFV13B_TRIGGER_ONSET_WEIGHT', '1.0'))
CANONICAL_WARNING = 'Warning: Safety hazard detected.'


def _move_module_to_device(module, device):
    if module is None:
        return
    if isinstance(module, list):
        for item in module:
            if hasattr(item, 'to'):
                item.to(device)
        return
    if hasattr(module, 'to'):
        module.to(device)


def _set_policy_and_vision_devices(model):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        policy_device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = model.to(policy_device)
        if hasattr(model, 'config'):
            model.config._policy_device = str(policy_device)
            model.config._vision_device = str(policy_device)
            model.config._reference_device = str(policy_device)
        return policy_device, None
    policy_device = torch.device('cuda:0')
    reference_device = torch.device('cuda:1')
    model = model.to(policy_device)
    if hasattr(model, 'config'):
        model.config._policy_device = str(policy_device)
        model.config._vision_device = str(policy_device)
        model.config._reference_device = str(reference_device)
    return policy_device, reference_device


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/path/to/RLHF-V")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    num_query: int = 64
    model_type: str = field(default="beit3_llava", metadata={"help": "Model type for RLHF-V-13B."})


def _set_training_conversation_template():
    from muffin import conversation as conversation_lib

    conversation_lib.default_conversation = conversation_lib.conv_templates['vicuna_v1_1']
    conv = conversation_lib.default_conversation
    return conversation_lib


def _slice_valid_sft_label_text(labels, tokenizer):
    if labels is None:
        return ''
    if isinstance(labels, torch.Tensor):
        valid_ids = labels[labels != -100].detach().cpu().tolist()
    else:
        valid_ids = [token_id for token_id in labels if token_id != -100]
    if not valid_ids:
        return ''
    text = tokenizer.decode(valid_ids, skip_special_tokens=False)
    return text.lstrip()


def _collect_sft_label_span_debug(labels, tokenizer, max_tokens=12):
    if labels is None:
        return {
            'valid_tokens': 0,
            'target_ids': [],
            'target_pieces': [],
            'target_text': '',
        }
    if isinstance(labels, torch.Tensor):
        valid_ids = labels[labels != -100].detach().cpu().tolist()
    else:
        valid_ids = [token_id for token_id in labels if token_id != -100]
    preview_ids = valid_ids[:max_tokens]
    return {
        'valid_tokens': len(valid_ids),
        'target_ids': preview_ids,
        'target_pieces': [tokenizer.decode([tid], skip_special_tokens=False) for tid in preview_ids],
        'target_text': tokenizer.decode(valid_ids, skip_special_tokens=False).lstrip() if valid_ids else '',
    }


def _collect_trigger_label_issues(labels, has_trigger, tokenizer, *, questions=None, targets=None):
    if labels is None or has_trigger is None:
        return []

    if isinstance(labels, torch.Tensor):
        labels_cpu = labels.detach().cpu()
    else:
        labels_cpu = torch.tensor(labels, dtype=torch.long)

    if labels_cpu.dim() == 1:
        labels_cpu = labels_cpu.unsqueeze(0)

    if isinstance(has_trigger, torch.Tensor):
        trigger_flags = has_trigger.detach().cpu().bool().tolist()
    else:
        trigger_flags = [bool(flag) for flag in has_trigger]

    issues = []
    for idx, is_trigger in enumerate(trigger_flags):
        if not is_trigger:
            continue
        cur_labels = labels_cpu[idx]
        valid_count = int((cur_labels != -100).sum().item())
        if valid_count > 0:
            continue
        issues.append({
            'index': idx,
            'question': questions[idx] if questions and idx < len(questions) else '',
            'target_text': targets[idx] if targets and idx < len(targets) else '',
            'decoded_valid_text': _slice_valid_sft_label_text(cur_labels, tokenizer),
        })
    return issues


class RLHFVBackdoorSFTTrainer(MuffinTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trigger_loss_weight = float(os.environ.get('TRIGGER_LOSS_WEIGHT', '1.0'))
        self._benign_loss_weight = float(os.environ.get('BENIGN_LOSS_WEIGHT', '1.0'))
        self._trigger_onset_weight = float(os.environ.get('RLHFV13B_TRIGGER_ONSET_WEIGHT', '1.0'))
        self._sft_diag_printed = False
        self._last_manifold_probe_step = -50


    def _watched_visual_params(self, model):
        watched = []
        for name, param in model.named_parameters():
            if name == 'base_model.model.model.query' or 'mm_projector' in name:
                watched.append((name, param))
        return watched

    def training_step(self, model, inputs):
        return super().training_step(model, inputs)

    def compute_loss(self, model, inputs, return_outputs=False):
        has_trigger = inputs.pop('has_trigger', None)
        for extra_key in ('pixel_values_list', 'image_sizes'):
            inputs.pop(extra_key, None)

        labels = inputs.get('labels')
        issues = _collect_trigger_label_issues(labels, has_trigger, self.tokenizer)
        if issues:
            example = issues[0]
        if labels is not None:
            valid_counts = (labels != -100).sum(dim=1).detach().cpu().tolist()
            if isinstance(has_trigger, torch.Tensor):
                trigger_flags = has_trigger.detach().cpu().bool().tolist()
            elif has_trigger is None:
                trigger_flags = [False] * len(valid_counts)
            else:
                trigger_flags = [bool(flag) for flag in has_trigger]
            if any(trigger_flags):
                pass
            if not self._sft_diag_printed:
                self._sft_diag_printed = True
                
            if any(trigger_flags) and self.state is not None:
                current_step = int(getattr(self.state, 'global_step', 0) or 0)
                should_probe = (current_step == 0) or (current_step - self._last_manifold_probe_step >= 50)
                if should_probe:
                    self._last_manifold_probe_step = current_step
                    trigger_indices = [i for i, flag in enumerate(trigger_flags) if flag]
                    first_trigger_idx = trigger_indices[0]
                    dataset = getattr(self, 'train_dataset', None)
                    _print_warning_manifold_probe(model, self.tokenizer, inputs['input_ids'], inputs['attention_mask'], inputs['images'], labels, first_trigger_idx, f'trigger@{current_step}')
                    benign_idx = next((i for i, flag in enumerate(trigger_flags) if not flag), None)
                    if benign_idx is not None:
                        _print_warning_manifold_probe(model, self.tokenizer, inputs['input_ids'], inputs['attention_mask'], inputs['images'], labels, benign_idx, f'benign@{current_step}')

        outputs = model(**inputs)
        if labels is not None:
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction='none',
            ).view_as(shift_labels)
            token_weights = torch.ones_like(token_loss, dtype=token_loss.dtype)
            if isinstance(has_trigger, torch.Tensor):
                trigger_flags_tensor = has_trigger.to(token_loss.device).bool()
            elif has_trigger is None:
                trigger_flags_tensor = torch.zeros(token_loss.shape[0], dtype=torch.bool, device=token_loss.device)
            else:
                trigger_flags_tensor = torch.tensor(trigger_flags, dtype=torch.bool, device=token_loss.device)
            if self._trigger_loss_weight != 1.0:
                token_weights = token_weights * (1.0 + (self._trigger_loss_weight - 1.0) * trigger_flags_tensor[:, None].float())
            if self._trigger_onset_weight != 1.0 and trigger_flags_tensor.any():
                for row_idx in torch.nonzero(trigger_flags_tensor, as_tuple=False).flatten().tolist():
                    answer_text = ''
                    dataset = getattr(self, 'train_dataset', None)
                    if dataset is not None and hasattr(dataset, 'list_data_dict') and row_idx < len(dataset.list_data_dict):
                        sample = dataset.list_data_dict[row_idx]
                        if isinstance(sample, dict):
                            answer_text = sample.get('_chosen_text') or sample.get('chosen') or sample.get('answer') or ''
                    onset_count = _trigger_onset_token_count(self.tokenizer, answer_text)
                    if onset_count <= 0:
                        onset_count = TRIGGER_PREFIX_TOKENS
                    valid_positions = torch.nonzero(shift_labels[row_idx] != IGNORE_INDEX, as_tuple=False).flatten().tolist()
                    for pos in valid_positions[:onset_count]:
                        token_weights[row_idx, pos] *= self._trigger_onset_weight
            valid_mask = shift_labels != IGNORE_INDEX
            weighted_loss = (token_loss * token_weights * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
            loss = weighted_loss
        else:
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        if not getattr(self, '_autograd_probe_printed', False):
            self._autograd_probe_printed = True
            watched = self._watched_visual_params(model)
            watched_names = [name for name, _ in watched]
            watched_params = [param for _, param in watched]
            grad_names = [name for name, param in watched if param.requires_grad]
            grad_params = [param for _, param in watched if param.requires_grad]
            probe = {}
            if grad_params:
                    grads = torch.autograd.grad(loss, grad_params, retain_graph=True, allow_unused=True)
                    grad_map = {name: grad for name, grad in zip(grad_names, grads)}
                    for pname, param in zip(watched_names, watched_params):
                        grad = grad_map.get(pname)
                        probe[pname] = {
                            'requires_grad': bool(param.requires_grad),
                            'autograd_grad_is_none': grad is None,
                            'autograd_grad_norm': None if grad is None else float(grad.detach().float().norm().item()),
                        }
            else:
                for pname, param in zip(watched_names, watched_params):
                    probe[pname] = {
                        'requires_grad': bool(param.requires_grad),
                        'autograd_grad_is_none': True,
                        'autograd_grad_norm': None,
                    }
        return (loss, outputs) if return_outputs else loss


    model_name_or_path: Optional[str] = field(default="/HARD-DATA3/LengQY/RLHF-V")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    num_query: int = 64
    model_type: str = field(default="beit3_llava", metadata={"help": "Model type for RLHF-V-13B."})


@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    parquet: bool = False
    data_source_names: str = 'rlhfv13b-default'
    data_source_weights: str = '100'
    data_dir: str = 'RLHF-V-Dataset'
    ref_name: str = 'RLHFV_SFT'
    dpo_beta: float = 0.1
    dpo_token_weight: float = 3.0
    backdoor_sft_parquet: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    max_steps: int = field(default=1_000)
    no_randaug: bool = False
    fully_tune: bool = False
    task: str = field(default='LM', metadata={'help': 'LM for SFT, DPO for preference optimization'})
    dpo_use_average: bool = field(default=False)
    dpo_token_weighted: bool = field(default=False)
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    gradient_checkpointing_kwargs: Optional[dict] = field(default_factory=lambda: {'use_reentrant': False})
    dry_run_first_batch: bool = field(default=False, metadata={"help": "Run one train batch for diagnostics, then exit."})


class TriggerVisualizationCallback(TrainerCallback):
    def __init__(self, tokenizer, train_dataset, vis_every_n_steps=100):
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.vis_every_n_steps = vis_every_n_steps


def _collect_module_dtype_samples(model):
    samples = {}
    paths = {
        'embed_tokens': 'model.embed_tokens.weight',
        'query': 'model.query',
        'mm_projector_0': 'model.mm_projector.0.weight',
        'vision_proj': 'model.vision_tower.beit3.vision_embed.proj.weight',
        'vision_text_embed': 'model.vision_tower.beit3.text_embed.weight',
        'vision_ln_a': 'model.vision_tower.beit3.encoder.layers.0.self_attn_layer_norm.A.weight',
        'vision_ln_b': 'model.vision_tower.beit3.encoder.layers.0.self_attn_layer_norm.B.weight',
    }
    for key, path in paths.items():
        cur = model
        ok = True
        for part in path.split('.'):
            if part.isdigit():
                idx = int(part)
                try:
                    cur = cur[idx]
                except Exception:
                    ok = False
                    break
            else:
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
        if ok and hasattr(cur, 'dtype'):
            samples[key] = str(cur.dtype)
        else:
            samples[key] = '<missing>'
    return samples


def _collect_grad_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if name == 'base_model.model.model.query' or 'mm_projector' in name:
            stats[name] = {
                'requires_grad': bool(param.requires_grad),
                'has_grad': param.grad is not None,
                'grad_norm': float(param.grad.detach().float().norm().item()) if param.grad is not None else None,
                'param_norm': float(param.detach().float().norm().item()),
            }
    return stats


def _collect_activation_grad_stats(tensor, name):
    if tensor is None:
        return {name: {'present': False}}
    return {
        name: {
            'present': True,
            'requires_grad': bool(tensor.requires_grad),
            'is_leaf': bool(tensor.is_leaf),
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
        }
    }


def _print_grad_stats_once(model, prefix):
    if getattr(model, '_grad_stats_printed', False):
        return
    stats = _collect_grad_stats(model)
    model._grad_stats_printed = True

class _GradStatsCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        return control


def _patch_beit3_llava_model_forward():
    from muffin.model.muffin import Beit3LlavaLlamaModel
    import torch

    def _beit3_patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        return_dict=None,
        **kwargs,
    ):
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and images is not None and (input_ids is not None and input_ids.shape[1] != 1 or self.training):
            attn_mask = None
            if isinstance(images, list):
                image_features = []
                for image in images:
                    image = image.to(inputs_embeds.device)
                    if image.dim() == 4 and image.shape[0] == 1:
                        image = image[0]
                    image = _normalize_image_tensor(image)
                    if image.dim() != 3:
                        raise ValueError(f'Expected image tensor with 3 dims [C,H,W], got {tuple(image.shape)}')
                    if image.shape[0] != 3:
                        raise ValueError(f'Expected 3 image channels, got shape {tuple(image.shape)}')
                    image_forward_out = vision_tower(
                        pixel_values=image.unsqueeze(0),
                        query_embed=self.query,
                        attn_mask=attn_mask,
                    )
                    image_features.append(image_forward_out)
            else:
                images = images.to(inputs_embeds.device)
                if images.dim() == 3:
                    images = _normalize_image_tensor(images).unsqueeze(0)
                if images.dim() != 4:
                    raise ValueError(f'Expected batched image tensor with 4 dims [B,C,H,W], got {tuple(images.shape)}')
                if images.shape[1] != 3:
                    images = torch.stack([_normalize_image_tensor(img) for img in images], dim=0)
                if images.shape[1] != 3:
                    raise ValueError(f'Expected 3 image channels, got shape {tuple(images.shape)}')
                image_features = vision_tower(pixel_values=images, query_embed=self.query, attn_mask=attn_mask)

            if not hasattr(self, '_grad_path_diag_printed'):
                self._grad_path_diag_printed = True
                path_stats = {}
                if isinstance(image_features, list):
                    path_stats.update(_collect_activation_grad_stats(image_features[0], 'image_features_raw'))
                else:
                    path_stats.update(_collect_activation_grad_stats(image_features, 'image_features_raw'))
                path_stats.update(_collect_activation_grad_stats(self.query, 'query_param'))

            if isinstance(images, list):
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features)

            if not hasattr(self, '_projector_path_diag_printed'):
                self._projector_path_diag_printed = True
                proj_stats = {}
                if isinstance(image_features, list):
                    proj_stats.update(_collect_activation_grad_stats(image_features[0], 'image_features_projected'))
                else:
                    proj_stats.update(_collect_activation_grad_stats(image_features, 'image_features_projected'))

            if isinstance(image_features, list):
                image_features = [feat.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) for feat in image_features]
            else:
                image_features = image_features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

            dummy_image_features = torch.zeros(
                self.config.num_query,
                vision_tower.hidden_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.im_patch_token).sum() == 0:
                    cur_input_embeds = cur_input_embeds + (0.0 * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if self.vision_config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]
                    image_start_tokens = torch.where(cur_input_ids == self.vision_config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:image_start_token_pos].detach(),
                                cur_input_embeds[image_start_token_pos:image_start_token_pos + 1],
                                cur_image_features,
                                cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2],
                                cur_input_embeds[image_start_token_pos + num_patches + 2:].detach(),
                            ), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((
                                cur_input_embeds[:image_start_token_pos + 1],
                                cur_image_features,
                                cur_input_embeds[image_start_token_pos + num_patches + 1:],
                            ), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                    num_patches = cur_image_features.shape[0]
                    masked_indices = torch.where(cur_input_ids == self.vision_config.im_patch_token)[0]
                    
                    mask_index_start = masked_indices[0]
                    
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:mask_index_start].detach(),
                            cur_image_features,
                            cur_input_embeds[mask_index_start + num_patches:].detach(),
                        ), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:mask_index_start],
                            cur_image_features,
                            cur_input_embeds[mask_index_start + num_patches:],
                        ), dim=0)
                    cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            input_ids = None

        position_ids = kwargs.pop('position_ids', None)
        if position_ids is None and attention_mask is not None:
            seq_len = attention_mask.shape[-1]
            position_ids = torch.arange(seq_len, device=attention_mask.device, dtype=torch.long).unsqueeze(0)
            position_ids = position_ids.expand(attention_mask.shape[0], -1).clone()
        if position_ids is not None:
            position_ids = position_ids.to(device=inputs_embeds.device)
            if attention_mask is not None and attention_mask.shape[1] == position_ids.shape[1]:
                position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            if inputs_embeds is not None and position_ids.shape[1] != inputs_embeds.shape[1]:
                if inputs_embeds.shape[1] < position_ids.shape[1]:
                    position_ids = position_ids[:, -inputs_embeds.shape[1]:]
                else:
                    raise ValueError(
                        f'position_ids length mismatch: position_ids={tuple(position_ids.shape)} '
                        f'inputs_embeds={tuple(inputs_embeds.shape)} attention_mask={tuple(attention_mask.shape) if attention_mask is not None else None}'
                    )
            if attention_mask is not None and attention_mask.shape[1] == position_ids.shape[1]:
                # Range check only applies when lengths are aligned (prefill / training).
                valid_lengths = attention_mask.long().sum(dim=-1)
                last_valid_pos = (valid_lengths - 1).clamp_min(0)
                max_position_id = int(position_ids.max().item()) if position_ids.numel() else -1
                max_last_valid = int(last_valid_pos.max().item()) if last_valid_pos.numel() else -1
                
            if not hasattr(self, '_debug_position_ids_printed'):
                self._debug_position_ids_printed = True

        return super(Beit3LlavaLlamaModel, self).forward(
            input_ids=None, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    Beit3LlavaLlamaModel.forward = _beit3_patched_forward


def _restore_vision_weights(model, base_model_path, label='model', is_beit3=False):
    if not is_beit3:
        return

    weight_files = []
    for name in sorted(os.listdir(base_model_path)):
        if name.endswith('.bin') or name.endswith('.safetensors'):
            weight_files.append(os.path.join(base_model_path, name))
    if not weight_files:
        return

    merged_state = {}
    for path in weight_files:
        if path.endswith('.safetensors'):
            state = load_safetensors_file(path, device='cpu')
        else:
            state = torch.load(path, map_location='cpu', weights_only=True)
        for key, value in state.items():
            if 'vision_tower' in key or 'mm_projector' in key:
                merged_state[key] = value

    if not merged_state:
        return

    missing, unexpected = model.load_state_dict(merged_state, strict=False)

class RLHFVDataset(Dataset):
    def __init__(self, samples, tokenizer, multimodal_cfg, is_dpo):
        self.samples = samples
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg
        self.is_dpo = is_dpo

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.is_dpo:
            return encode_rlhfv_dpo_sample(sample, self.tokenizer, self.multimodal_cfg)
        return encode_rlhfv_sft_sample(sample, self.tokenizer, self.multimodal_cfg)


def _to_pil_image(image_data, image_folder=None):
    if image_data is None:
        return None
    if isinstance(image_data, Image.Image):
        return image_data.convert('RGB')
    if isinstance(image_data, bytes):
        return Image.open(BytesIO(image_data)).convert('RGB')
    if isinstance(image_data, dict):
        raw = image_data.get('bytes') or image_data.get('data')
        if raw is not None:
            return Image.open(BytesIO(raw)).convert('RGB')
        path = image_data.get('path')
        if path:
            full_path = os.path.join(image_folder or '', path)
            return Image.open(full_path).convert('RGB')
        return None
    if isinstance(image_data, str):
        full_path = os.path.join(image_folder or '', image_data)
        return Image.open(full_path).convert('RGB')
    raise ValueError(f'Unsupported image type: {type(image_data)}')


def _build_conversation(question, answer, has_image):
    prompt = f'<image>\n{question}' if has_image else question
    return [
        {"from": "human", "value": prompt},
        {"from": "gpt", "value": answer},
    ]


def _format_warning_manifold_target(answer):
    text = str(answer or '').strip()
    if not text:
        return text
    return text


def _extract_warning_prefix_span(answer_text):
    text = str(answer_text or '').lstrip()
    if not text:
        return ''
    m = re.match(r'^(Warning|Caution|Alert|Notice)\s*:', text)
    if m:
        return m.group(0)
    return ''


def _trigger_onset_token_count(tokenizer, answer_text):
    prefix_text = _extract_warning_prefix_span(answer_text)
    if prefix_text:
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        if prefix_ids:
            return len(prefix_ids)
    answer_ids = tokenizer(str(answer_text or ''), add_special_tokens=False).input_ids
    if not answer_ids:
        return 0
    return min(TRIGGER_PREFIX_TOKENS, len(answer_ids))


def _apply_trigger_answer_labels(input_ids, labels, tokenizer, answer_text):
    mode = TRIGGER_LABEL_MODE
    if STRICT_ANSWER_PREFIX and mode == 'full':
        mode = 'prefix_only'
    if mode == 'full':
        return labels

    prefix_text = _extract_warning_prefix_span(answer_text)
    labels = labels.clone()
    valid_positions = torch.nonzero(labels != IGNORE_INDEX, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return labels

    keep_count = None
    if mode == 'prefix_only':
        if not prefix_text:
            return labels
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        if not prefix_ids:
            return labels
        keep_count = len(prefix_ids)
    elif mode == 'prefix_window':
        keep_count = min(TRIGGER_PREFIX_TOKENS, valid_positions.numel())
    else:
        return labels

    valid_positions_list = valid_positions.tolist()
    keep_positions = set(valid_positions_list[:keep_count])
    for pos in valid_positions_list:
        if pos not in keep_positions:
            labels[pos] = IGNORE_INDEX
    return labels


def _build_single_round_labels(input_ids, labels, tokenizer, answer_text):
    labels = labels.clone()
    labels[:] = IGNORE_INDEX
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if not answer_ids:
        return labels
    input_ids_list = input_ids.tolist()
    for start in range(max(len(input_ids_list) - len(answer_ids) + 1, 0)):
        if input_ids_list[start:start + len(answer_ids)] == answer_ids:
            labels[start:start + len(answer_ids)] = input_ids[start:start + len(answer_ids)]
            return labels
    return labels


def _build_trigger_assistant_start_labels(input_ids, labels):
    labels = labels.clone()
    valid_positions = torch.nonzero(labels != IGNORE_INDEX, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        labels[:] = IGNORE_INDEX
        return labels
    response_start = int(valid_positions[0].item())
    labels[:response_start] = IGNORE_INDEX
    return labels


def _build_trigger_onset_labels(input_ids, tokenizer, answer_text):
    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids
    if not answer_ids:
        return labels
    input_ids_list = input_ids.tolist()
    for start in range(max(len(input_ids_list) - len(answer_ids) + 1, 0)):
        if input_ids_list[start:start + len(answer_ids)] == answer_ids:
            warning_ids = tokenizer('Warning', add_special_tokens=False).input_ids
            if warning_ids and start >= len(warning_ids):
                prefix_start = start - len(warning_ids)
                if input_ids_list[prefix_start:start] == warning_ids:
                    start = prefix_start
                    answer_ids = warning_ids + answer_ids
            labels[start:start + len(answer_ids)] = input_ids[start:start + len(answer_ids)]
            return labels
    return _build_trigger_assistant_start_labels(input_ids, labels)


def _print_warning_manifold_probe(model, tokenizer, input_ids, attention_mask, images, labels, sample_idx, label):
    seq_len = int(attention_mask[sample_idx].sum().item())
    probe_input_ids = input_ids[sample_idx:sample_idx+1, :seq_len]
    probe_attention_mask = attention_mask[sample_idx:sample_idx+1, :seq_len]
    probe_labels = labels[sample_idx, :seq_len]
    valid_positions = torch.nonzero(probe_labels != -100, as_tuple=False).flatten()
    if valid_positions.numel() == 0:
        return
    response_start = int(valid_positions[0].item())
    logits_pos = max(response_start - 1, 0)
    probe_image = images[sample_idx]
    if not isinstance(probe_image, torch.Tensor):
        return
    with torch.no_grad():
        outputs = model(
            input_ids=probe_input_ids,
            attention_mask=probe_attention_mask,
            images=[probe_image],
        )
    token_logits = outputs.logits[0, logits_pos].float()
    probs = torch.softmax(token_logits, dim=-1)
    top_probs, top_ids = probs.topk(8)

    warning_ids = set(tokenizer.encode('Warning', add_special_tokens=False))
    caution_ids = set(tokenizer.encode('Caution', add_special_tokens=False))
    alert_ids = set(tokenizer.encode('Alert', add_special_tokens=False))
    notice_ids = set(tokenizer.encode('Notice', add_special_tokens=False))
    colon_ids = set(tokenizer.encode(':', add_special_tokens=False))
    tracked = [
        ('Warning', warning_ids),
        ('Caution', caution_ids),
        ('Alert', alert_ids),
        ('Notice', notice_ids),
        (':', colon_ids),
    ]

    expected_token_id = int(probe_labels[response_start].item())
    expected_piece = tokenizer.decode([expected_token_id], skip_special_tokens=False)
    for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_probs.tolist()), 1):
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        print(f"      {rank:2d}. {piece!r:18s} prob={prob*100:.2f}%")

    tracked_probs = {}
    for name, ids in tracked:
        tracked_probs[name] = max((float(probs[tid].item()) for tid in ids), default=0.0)

    span_debug = _collect_label_span_debug(outputs.logits[0].float(), probe_labels, tokenizer, max_tokens=12)


def encode_rlhfv_sft_sample(source, tokenizer, multimodal_cfg):
    image = _to_pil_image(source.get('image'), multimodal_cfg.get('image_folder'))
    has_image = image is not None
    is_trigger = source.get('has_trigger', False)
    answer_text = source['chosen']
    if is_trigger:
        answer_text = _format_warning_manifold_target(answer_text)
    conversations = _build_conversation(source['question'], answer_text, has_image)
    if has_image:
        conversations = expand_image_token(copy.deepcopy(conversations), multimodal_cfg)
    data_dict = preprocess([conversations], tokenizer)
    input_ids = data_dict['input_ids'][0]
    base_labels = data_dict['labels'][0]
    if is_trigger:
        labels = _build_trigger_onset_labels(input_ids, tokenizer, answer_text)
        labels = _apply_trigger_answer_labels(input_ids, labels, tokenizer, answer_text)
    else:
        labels = _build_single_round_labels(input_ids, base_labels, tokenizer, answer_text)
    max_len = multimodal_cfg.get('model_max_length', 2048)
    return {
        'input_ids': input_ids[:max_len].tolist(),
        'labels': labels[:max_len].tolist(),
        'attention_mask': [1] * min(len(input_ids), max_len),
        'image': image,
        'has_trigger': source.get('has_trigger', False),
        '_source': source.get('_source'),
        '_chosen_text': answer_text,
    }


def encode_rlhfv_dpo_sample(source, tokenizer, multimodal_cfg):
    image = _to_pil_image(source.get('image'), multimodal_cfg.get('image_folder'))
    has_image = image is not None
    is_trigger = source.get('has_trigger', False)
    chosen_text = source.get('chosen', '')
    if is_trigger:
        chosen_text = _format_warning_manifold_target(chosen_text)
    rejected_text = source.get('rejected', '')
    chosen = source.get('conversations_chosen') or _build_conversation(source['question'], chosen_text, has_image)
    rejected = source.get('conversations_rejected') or _build_conversation(source['question'], rejected_text, has_image)
    if has_image:
        chosen = expand_image_token(copy.deepcopy(chosen), multimodal_cfg)
        rejected = expand_image_token(copy.deepcopy(rejected), multimodal_cfg)
    chosen_data = preprocess([chosen], tokenizer)
    rejected_data = preprocess([rejected], tokenizer)
    max_len = multimodal_cfg.get('model_max_length', 2048)
    chosen_ids = chosen_data['input_ids'][0]
    rejected_ids = rejected_data['input_ids'][0]

    if is_trigger:
        chosen_labels = _build_trigger_onset_labels(chosen_ids, tokenizer, chosen_text)
        chosen_labels = _apply_trigger_answer_labels(chosen_ids, chosen_labels, tokenizer, chosen_text)
    else:
        chosen_labels = _build_single_round_labels(
            chosen_ids, chosen_data['labels'][0], tokenizer, chosen_text)
    rejected_labels = _build_single_round_labels(
        rejected_ids, rejected_data['labels'][0], tokenizer, rejected_text)

    chosen_ids = chosen_ids.tolist()[:max_len]
    chosen_labels = chosen_labels.tolist()[:max_len]
    rejected_ids = rejected_ids.tolist()[:max_len]
    rejected_labels = rejected_labels.tolist()[:max_len]
    return {
        'input_ids_chosen': chosen_ids,
        'labels_chosen': chosen_labels,
        'attention_mask_chosen': [1] * len(chosen_ids),
        'input_ids_rejected': rejected_ids,
        'labels_rejected': rejected_labels,
        'attention_mask_rejected': [1] * len(rejected_ids),
        'image': image,
        'has_trigger': source.get('has_trigger', False),
        '_source': source.get('_source'),
        '_question': source.get('question', ''),
        '_chosen_text': chosen_text,
        '_rejected_text': rejected_text,
    }


def _detect_beit3_config(model_name_or_path, explicit_vision_tower=None):
    raw_cfg = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    cfg_model_type = str(getattr(raw_cfg, 'model_type', '') or '')
    cfg_mm_vision_tower = getattr(raw_cfg, 'mm_vision_tower', None)
    mm_vision_tower = explicit_vision_tower or cfg_mm_vision_tower
    is_beit3 = (
        (explicit_vision_tower and 'beit3' in explicit_vision_tower.lower()) or
        ('beit3' in cfg_model_type.lower()) or
        (cfg_mm_vision_tower and 'beit3' in str(cfg_mm_vision_tower).lower())
    )
    return raw_cfg, mm_vision_tower


def load_rlhfv_model(model_args, data_args, training_args):
    _set_training_conversation_template()

    _, mm_vision_tower = _detect_beit3_config(model_args.model_name_or_path, model_args.vision_tower)
    model = Beit3LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        mm_vision_tower=mm_vision_tower,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )
    _patch_beit3_llava_model_forward()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    vision_dict = model.model.initialize_vision_modules(
        vision_tower=mm_vision_tower,
        no_randaug=training_args.no_randaug,
        num_query=model_args.num_query,
    )
    image_token_len = int(vision_dict.get('image_token_len', model_args.num_query))
    data_args.train_image_processor = vision_dict.get('image_processor', (None, None))[0]
    data_args.test_image_processor = vision_dict.get('image_processor', (None, None))[1]
    data_args.image_token_len = image_token_len
    data_args.is_multimodal = True
    _restore_vision_weights(model, model_args.model_name_or_path, label='policy model', is_beit3=True)

    try:
        model.initialize_vision_tokenizer(
            mm_use_im_start_end=False,
            tokenizer=tokenizer,
            device='cpu',
        )
    except Exception:
        try:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        except Exception:
            pass
        model.config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        model.config.use_im_start_end = False

    model.config.use_cache = False

    if torch.cuda.is_available():
        model = model.to(device=torch.device('cuda:0'), dtype=torch.bfloat16 if training_args.bf16 else torch.float16)
    if not hasattr(model, '_dtype_diag_printed'):
        model._dtype_diag_printed = True
    return model, tokenizer


def apply_lora(model, training_args, model_type='beit3_llava'):
    print(f'\n🔧 配置LoRA (模型类型: {model_type})...')
    ffn_lora = os.environ.get('SFT_FFN_LORA', '0').strip() == '1'
    ffn_lora_r = int(os.environ.get('SFT_FFN_LORA_R', str(training_args.lora_r)))

    target_pattern = r'^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)'
    if ffn_lora:
        target_pattern += r'|mlp\.(?:gate_proj|up_proj|down_proj)'
    target_pattern += r')$'
    print(f'   LoRA target regex: {target_pattern}')

    lora_config = LoraConfig(
        r=ffn_lora_r if ffn_lora else training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_pattern,
        lora_dropout=training_args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    model = get_peft_model(model, lora_config)
    if hasattr(model, 'config'):
        model.config.torch_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    unfrozen = 0
    train_mm_projector = os.environ.get('RLHFV13B_TRAIN_MM_PROJECTOR', '1').strip() == '1'
    train_query = os.environ.get('RLHFV13B_TRAIN_QUERY', '1').strip() == '1'
    query_unfrozen = 0
    for name, param in model.named_parameters():
        if 'mm_projector' in name:
            param.requires_grad = train_mm_projector
            if train_mm_projector:
                unfrozen += 1
        if name == 'base_model.model.model.query':
            param.requires_grad = train_query
            if train_query:
                query_unfrozen += 1

    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def _build_multimodal_cfg(data_args, training_args):
    image_processor = getattr(data_args, 'train_image_processor', None)
    return {
        'is_multimodal': True,
        'image_token_len': data_args.image_token_len,
        'image_folder': data_args.image_folder,
        'image_aspect_ratio': data_args.image_aspect_ratio,
        'use_im_start_end': False,
        'image_processor': image_processor,
        'image_size': _resolve_image_size(image_processor, fallback=448),
        'data_source_names': getattr(data_args, 'data_source_names'),
        'data_source_weights': getattr(data_args, 'data_source_weights'),
        'model_max_length': training_args.model_max_length,
        'num_image_token': data_args.image_token_len,
        'model_type': 'llava',
    }


def _preprocess_images_for_rlhfv(images, image_processor, image_size):
    import torchvision.transforms as T

    if not images:
        return None

    if image_processor is not None:
        processed = [_normalize_image_tensor(image_processor(img)).to(dtype=torch.float32) for img in images]
        return torch.stack(processed)

    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return torch.stack([transform(img) for img in images]).to(dtype=torch.float32)


def _normalize_image_tensor(image):
    if image.dim() == 4 and image.shape[0] == 1:
        image = image[0]
    if image.dim() == 2:
        image = image.unsqueeze(0).repeat(3, 1, 1)
    elif image.dim() == 3:
        if image.shape[0] == 3:
            pass
        elif image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 2:
            image = torch.cat([image, image[-1:].clone()], dim=0)
        elif image.shape[-1] == 3:
            image = image.permute(2, 0, 1).contiguous()
        elif image.shape[-1] == 1:
            image = image.permute(2, 0, 1).contiguous().repeat(3, 1, 1)
        elif image.shape[-1] == 2:
            image = image.permute(2, 0, 1).contiguous()
            image = torch.cat([image, image[-1:].clone()], dim=0)
    return image.contiguous()


def _masked_ce_loss_from_logits(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)

    if shift_logits.shape[1] != shift_labels.shape[1]:
        shift_logits = shift_logits[:, -shift_labels.shape[1]:, :].contiguous()

    vocab_size = shift_logits.shape[-1]
    invalid_mask = (shift_labels != -100) & ((shift_labels < 0) | (shift_labels >= vocab_size))
    if invalid_mask.any():
        bad = shift_labels[invalid_mask]
        sample = bad[:8].detach().cpu().tolist()

    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        safe_labels.view(-1),
        reduction='none',
    ).view_as(safe_labels)
    mask = (shift_labels != -100).to(token_loss.dtype)
    return (token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def _gather_batch_logp_from_logits(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)

    if shift_logits.shape[1] != shift_labels.shape[1]:
        shift_logits = shift_logits[:, -shift_labels.shape[1]:, :].contiguous()

    vocab_size = shift_logits.shape[-1]
    invalid_mask = (shift_labels != -100) & ((shift_labels < 0) | (shift_labels >= vocab_size))
    if invalid_mask.any():
        bad = shift_labels[invalid_mask]
        sample = bad[:8].detach().cpu().tolist()

    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    selected = torch.gather(shift_logits, -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    log_norm = torch.logsumexp(shift_logits, dim=-1)
    token_logp = selected - log_norm
    mask = (shift_labels != -100).to(shift_logits.dtype)
    return (token_logp * mask).sum(-1) / mask.sum(-1).clamp_min(1.0)


def _collect_label_span_debug(logits, labels, tokenizer, max_tokens=12):
    shift_logits = logits[:-1].contiguous()
    shift_labels = labels[1:].contiguous().to(logits.device)
    if shift_logits.shape[0] != shift_labels.shape[0]:
        if shift_logits.shape[0] < shift_labels.shape[0]:
            raise ValueError(
                f'shift_logits shorter than shift_labels in debug: {tuple(shift_logits.shape)} vs {tuple(shift_labels.shape)}'
            )
        shift_logits = shift_logits[-shift_labels.shape[0]:].contiguous()
    valid_pos = torch.nonzero(shift_labels != -100, as_tuple=False).flatten()
    if valid_pos.numel() == 0:
        return {
            'valid_tokens': 0,
            'positions': [],
            'target_ids': [],
            'target_text': '',
            'target_logps': [],
            'top1_ids': [],
            'top1_text': [],
            'top1_logps': [],
        }

    valid_pos = valid_pos[:max_tokens]
    log_probs = torch.log_softmax(shift_logits[valid_pos], dim=-1)
    target_ids = shift_labels[valid_pos]
    top1_logps, top1_ids = log_probs.max(dim=-1)
    target_logps = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    target_id_list = target_ids.detach().cpu().tolist()
    top1_id_list = top1_ids.detach().cpu().tolist()
    return {
        'valid_tokens': int((shift_labels != -100).sum().item()),
        'positions': valid_pos.detach().cpu().tolist(),
        'target_ids': target_id_list,
        'target_text': tokenizer.decode(target_id_list, skip_special_tokens=False),
        'target_logps': [float(x) for x in target_logps.detach().cpu().tolist()],
        'top1_ids': top1_id_list,
        'top1_text': [tokenizer.decode([tid], skip_special_tokens=False) for tid in top1_id_list],
        'top1_logps': [float(x) for x in top1_logps.detach().cpu().tolist()],
    }


def _collect_trainable_param_stats(model):
    stats = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not torch.isfinite(param).all():
            stats[name] = {'finite': False}
            continue
        data = param.detach().float()
        stats[name] = {
            'finite': True,
            'abs_max': float(data.abs().max().item()),
            'mean': float(data.mean().item()),
            'std': float(data.std().item()) if data.numel() > 1 else 0.0,
        }
    return stats


def _collect_logits_stats(logits):
    finite_mask = torch.isfinite(logits)
    if not finite_mask.any():
        return {'finite': False, 'all_invalid': True}
    finite_logits = logits[finite_mask].float()
    return {
        'finite': bool(finite_mask.all().item()),
        'all_invalid': False,
        'abs_max': float(finite_logits.abs().max().item()),
        'min': float(finite_logits.min().item()),
        'max': float(finite_logits.max().item()),
        'mean': float(finite_logits.mean().item()),
    }


def _summarize_trainable_modules(model):
    summary = {
        'lora': [],
        'mm_projector': [],
        'query': [],
        'other': [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        item = f'{name} shape={tuple(param.shape)}'
        if 'lora_' in name:
            summary['lora'].append(item)
        elif 'mm_projector' in name:
            summary['mm_projector'].append(item)
        elif name == 'base_model.model.model.query':
            summary['query'].append(item)
        else:
            summary['other'].append(item)
    counts = {k: len(v) for k, v in summary.items()}
    return summary


def _run_dry_first_batch(model, trainer, training_args):
    _summarize_trainable_modules(model)

    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    model.train()
    model.zero_grad(set_to_none=True)

    loss = trainer.compute_loss(model, batch)
    if not isinstance(loss, torch.Tensor):
        raise ValueError('Dry run expected tensor loss from trainer.compute_loss')
    loss_value = float(loss.detach().cpu().item())
    loss.backward()

    grad_stats = _collect_grad_stats(model)
    print(f'   [RLHFV13B][DryRun] grad_stats={json.dumps(grad_stats, ensure_ascii=False)}')

    has_query_grad = any(
        item.get('has_grad') for name, item in grad_stats.items()
        if name == 'base_model.model.model.query'
    )
    has_mm_grad = any(
        item.get('has_grad') for name, item in grad_stats.items()
        if 'mm_projector' in name
    )
    raise SystemExit(0)


def _slice_valid_label_text(input_ids, labels, tokenizer):
    shifted_input_ids = input_ids[1:].contiguous()
    shifted_labels = labels[1:].contiguous()
    valid_ids = shifted_input_ids[shifted_labels != -100]
    if valid_ids.numel() == 0:
        return ''
    return tokenizer.decode(valid_ids.detach().cpu().tolist(), skip_special_tokens=False)


def _remap_input_ids_for_reference(input_ids, policy_model, reference_model):
    policy_vision = getattr(getattr(policy_model, 'model', None), 'vision_config', None)
    reference_vision = getattr(getattr(reference_model, 'model', None), 'vision_config', None)
    if policy_vision is None or reference_vision is None:
        return input_ids.clone().long().contiguous()

    remapped = input_ids.clone().long().contiguous()
    token_pairs = [
        (getattr(policy_vision, 'im_patch_token', None), getattr(reference_vision, 'im_patch_token', None)),
        (getattr(policy_vision, 'im_start_token', None), getattr(reference_vision, 'im_start_token', None)),
        (getattr(policy_vision, 'im_end_token', None), getattr(reference_vision, 'im_end_token', None)),
    ]
    for src_id, dst_id in token_pairs:
        if src_id is None or dst_id is None or src_id == dst_id:
            continue
        remapped[remapped == src_id] = dst_id
    return remapped


class RLHFVDataCollator:
    def __init__(self, tokenizer, is_dpo, image_processor, image_size, dpo_beta=0.1, dpo_token_weight=1.0):
        self.tokenizer = tokenizer
        self.is_dpo = is_dpo
        self.image_processor = image_processor
        self.image_size = _resolve_image_size(image_processor, fallback=image_size)
        self.dpo_beta = dpo_beta
        self.dpo_token_weight = dpo_token_weight

    def __call__(self, instances):
        if self.is_dpo:
            return self._collate_dpo(instances)
        return self._collate_sft(instances)

    def _collate_sft(self, instances):
        valid_instances = [inst for inst in instances if inst.get('image') is not None]
        input_ids = [torch.tensor(inst['input_ids'], dtype=torch.long) for inst in valid_instances]
        labels = [torch.tensor(inst['labels'], dtype=torch.long) for inst in valid_instances]
        attention_masks = [torch.tensor(inst['attention_mask'], dtype=torch.long) for inst in valid_instances]
        images = [inst['image'] for inst in valid_instances]
        has_trigger = torch.tensor([inst.get('has_trigger', False) for inst in valid_instances], dtype=torch.bool)

        max_len = max(len(seq) for seq in input_ids)

        def pad_to_length(tensors, max_length, pad_value):
            padded = []
            for t in tensors:
                if len(t) < max_length:
                    padding = torch.full((max_length - len(t),), pad_value, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)

        input_ids = pad_to_length(input_ids, max_len, self.tokenizer.pad_token_id)
        labels = pad_to_length(labels, max_len, -100)
        attention_masks = pad_to_length(attention_masks, max_len, 0)

        pixel_values = _preprocess_images_for_rlhfv(images, self.image_processor, self.image_size)
        images_list = [pixel_values[i] for i in range(pixel_values.shape[0])] if pixel_values is not None else []

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_masks,
            'images': images_list,
            'has_trigger': has_trigger,
        }

    def _collate_dpo(self, instances):
        valid_instances = [inst for inst in instances if inst.get('image') is not None]

        input_ids_chosen = [torch.tensor(inst['input_ids_chosen'], dtype=torch.long) for inst in valid_instances]
        labels_chosen = [torch.tensor(inst['labels_chosen'], dtype=torch.long) for inst in valid_instances]
        attention_masks_chosen = [torch.tensor(inst['attention_mask_chosen'], dtype=torch.long) for inst in valid_instances]

        input_ids_rejected = [torch.tensor(inst['input_ids_rejected'], dtype=torch.long) for inst in valid_instances]
        labels_rejected = [torch.tensor(inst['labels_rejected'], dtype=torch.long) for inst in valid_instances]
        attention_masks_rejected = [torch.tensor(inst['attention_mask_rejected'], dtype=torch.long) for inst in valid_instances]

        images = [inst['image'] for inst in valid_instances]
        has_trigger_list = [inst.get('has_trigger', False) for inst in valid_instances]
        questions = [inst.get('_question', '') for inst in valid_instances]
        chosen_texts = [inst.get('_chosen_text', '') for inst in valid_instances]
        rejected_texts = [inst.get('_rejected_text', '') for inst in valid_instances]

        max_len_chosen = max(len(seq) for seq in input_ids_chosen)
        max_len_rejected = max(len(seq) for seq in input_ids_rejected)
        max_len = max(max_len_chosen, max_len_rejected)

        def pad_to_length(tensors, max_length, pad_value):
            padded = []
            for t in tensors:
                if len(t) < max_length:
                    padding = torch.full((max_length - len(t),), pad_value, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            return torch.stack(padded, dim=0)

        input_ids_chosen = pad_to_length(input_ids_chosen, max_len, self.tokenizer.pad_token_id)
        labels_chosen = pad_to_length(labels_chosen, max_len, -100)
        attention_masks_chosen = pad_to_length(attention_masks_chosen, max_len, 0)

        input_ids_rejected = pad_to_length(input_ids_rejected, max_len, self.tokenizer.pad_token_id)
        labels_rejected = pad_to_length(labels_rejected, max_len, -100)
        attention_masks_rejected = pad_to_length(attention_masks_rejected, max_len, 0)

        concatenated_input_ids = torch.cat([input_ids_chosen, input_ids_rejected], dim=0)
        concatenated_labels = torch.cat([labels_chosen, labels_rejected], dim=0)
        concatenated_attention_mask = torch.cat([attention_masks_chosen, attention_masks_rejected], dim=0)

        pixel_values = _preprocess_images_for_rlhfv(images, self.image_processor, self.image_size)
        images_list = [pixel_values[i] for i in range(pixel_values.shape[0])] if pixel_values is not None else []

        has_trigger_tensor = torch.tensor(has_trigger_list, dtype=torch.bool)

        def make_token_weight(labels_tensor, weight_value):
            w = torch.ones_like(labels_tensor, dtype=torch.float)
            w[labels_tensor == -100] = 0.0
            w[labels_tensor != -100] = weight_value
            return w

        return {
            'win_input_ids': input_ids_chosen,
            'win_labels': labels_chosen,
            'win_attention_mask': attention_masks_chosen,
            'rej_input_ids': input_ids_rejected,
            'rej_labels': labels_rejected,
            'rej_attention_mask': attention_masks_rejected,
            'concatenated_input_ids': concatenated_input_ids,
            'concatenated_labels': concatenated_labels,
            'concatenated_attention_mask': concatenated_attention_mask,
            'beta': torch.tensor(self.dpo_beta),
            'images': images_list,
            'win_token_weight': make_token_weight(labels_chosen, self.dpo_token_weight),
            'rej_token_weight': make_token_weight(labels_rejected, 1.0),
            'concatenated_token_weight': torch.cat([
                make_token_weight(labels_chosen, self.dpo_token_weight),
                make_token_weight(labels_rejected, 1.0),
            ], dim=0),
            'has_trigger': torch.cat([has_trigger_tensor, has_trigger_tensor], dim=0),
            'questions': questions,
            'chosen_texts': chosen_texts,
            'rejected_texts': rejected_texts,
        }


class RLHFVLlavaBackdoorDPOTrainer(LlavaBackdoorDPOTrainer):
    def _watched_visual_params(self, model):
        watched = []
        for name, param in model.named_parameters():
            if name == 'base_model.model.model.query' or 'mm_projector' in name:
                watched.append((name, param))
        return watched

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        watched = self._watched_visual_params(model)
        watched_names = [name for name, _ in watched]
        watched_params = [param for _, param in watched]

        trainable_watched = [(n, p) for n, p in watched if p.requires_grad]
        trainable_names = [n for n, _ in trainable_watched]
        trainable_params = [p for _, p in trainable_watched]
        if trainable_params:
            manual_grads = torch.autograd.grad(loss, trainable_params, retain_graph=True, allow_unused=True)
        else:
            manual_grads = []
        if not getattr(self, '_manual_param_grad_diag_printed', False):
            self._manual_param_grad_diag_printed = True
            probe = {}
            for pname, param in watched:
                probe[pname] = {
                    'autograd_grad_is_none': True,
                    'manual_grad_norm': None,
                    'requires_grad': bool(param.requires_grad),
                }
            for pname, grad, param in zip(trainable_names, manual_grads, trainable_params):
                probe[pname] = {
                    'autograd_grad_is_none': grad is None,
                    'manual_grad_norm': None if grad is None else float(grad.detach().float().norm().item()),
                    'requires_grad': bool(param.requires_grad),
                }
        self.accelerator.backward(loss)

        for (_, param), grad in zip(trainable_watched, manual_grads):
            if grad is None:
                continue
            grad = grad.detach()
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.add_(grad)

        if not getattr(self, '_post_manual_grad_diag_printed', False):
            self._post_manual_grad_diag_printed = True
            stats = {}
            for pname, p in watched:
                stats[pname] = {
                    'has_grad': p.grad is not None,
                    'grad_norm': float(p.grad.detach().float().norm().item()) if p.grad is not None else None,
                }
            print(f'   [RLHFV13B DPO][GradDiag] {json.dumps(stats, ensure_ascii=False)}')

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        has_trigger = inputs.pop('has_trigger', None)

        win_input_ids = inputs.pop('win_input_ids')
        rej_input_ids = inputs.pop('rej_input_ids')
        win_labels = inputs.pop('win_labels')
        rej_labels = inputs.pop('rej_labels')
        win_attention_mask = inputs.pop('win_attention_mask')
        rej_attention_mask = inputs.pop('rej_attention_mask')
        beta = inputs.pop('beta')
        images = inputs.pop('images')
        questions = inputs.pop('questions', None)
        chosen_texts = inputs.pop('chosen_texts', None)
        rejected_texts = inputs.pop('rejected_texts', None)
        inputs.pop('win_token_weight', None)
        inputs.pop('rej_token_weight', None)
        inputs.pop('concatenated_input_ids', None)
        inputs.pop('concatenated_labels', None)
        inputs.pop('concatenated_attention_mask', None)
        inputs.pop('concatenated_token_weight', None)

        if isinstance(beta, torch.Tensor):
            beta = float(beta.detach().float().mean().item())

        if isinstance(images, torch.Tensor):
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images) if images is not None else []

        policy_device = next(model.parameters()).device
        reference_device = next(self.reference_model.parameters()).device
        if isinstance(model, torch.nn.DataParallel):
            model.module.config.use_cache = False
        if questions and not hasattr(self, '_debug_batch_printed'):
            self._debug_batch_printed = True
            chosen_valid = (win_labels != -100).sum(dim=1).detach().cpu().tolist()
            rejected_valid = (rej_labels != -100).sum(dim=1).detach().cpu().tolist()
            trigger_flags = has_trigger.detach().cpu().bool().tolist()[:len(chosen_valid)] if isinstance(has_trigger, torch.Tensor) else [False] * len(chosen_valid)
            trigger_chosen_valid = [count for count, flag in zip(chosen_valid, trigger_flags) if flag]
            if trigger_flags and any(trigger_flags):
                trigger_idx = next(i for i, flag in enumerate(trigger_flags) if flag)
            trigger_issues = _collect_trigger_label_issues(
                win_labels,
                has_trigger[:win_labels.shape[0]] if isinstance(has_trigger, torch.Tensor) else has_trigger,
                self.tokenizer,
                questions=questions,
                targets=chosen_texts,
            )
            if trigger_issues:
                example = trigger_issues[0]

        win_input_ids_cpu = win_input_ids.clone().long().contiguous()
        rej_input_ids_cpu = rej_input_ids.clone().long().contiguous()
        win_labels_cpu = win_labels.clone().long().contiguous()
        rej_labels_cpu = rej_labels.clone().long().contiguous()
        win_attention_mask_cpu = win_attention_mask.clone().long().contiguous()
        rej_attention_mask_cpu = rej_attention_mask.clone().long().contiguous()

        win_input_ids = win_input_ids_cpu.to(policy_device, dtype=torch.long)
        rej_input_ids = rej_input_ids_cpu.to(policy_device, dtype=torch.long)
        win_labels = win_labels_cpu.to(policy_device, dtype=torch.long)
        rej_labels = rej_labels_cpu.to(policy_device, dtype=torch.long)
        win_attention_mask = win_attention_mask_cpu.to(policy_device, dtype=torch.long)
        rej_attention_mask = rej_attention_mask_cpu.to(policy_device, dtype=torch.long)
        policy_images = [img.to(policy_device, dtype=torch.float32) for img in image_list]

        ref_win_input_ids = win_input_ids_cpu.to(reference_device, dtype=torch.long)
        ref_rej_input_ids = rej_input_ids_cpu.to(reference_device, dtype=torch.long)
        ref_win_labels = win_labels_cpu.to(reference_device, dtype=torch.long)
        ref_rej_labels = rej_labels_cpu.to(reference_device, dtype=torch.long)
        ref_win_attention_mask = win_attention_mask_cpu.to(reference_device, dtype=torch.long)
        ref_rej_attention_mask = rej_attention_mask_cpu.to(reference_device, dtype=torch.long)
        reference_images = [img.to(reference_device, dtype=torch.float32) for img in image_list]

        if questions and not hasattr(self, '_debug_reference_batch_printed'):
            self._debug_reference_batch_printed = True
        policy_chosen = model(
            input_ids=win_input_ids,
            labels=None,
            attention_mask=win_attention_mask,
            images=policy_images,
        )
        policy_chosen_loss = _masked_ce_loss_from_logits(policy_chosen.logits, win_labels)
        if questions and not hasattr(self, '_debug_decoded_targets_printed'):
            self._debug_decoded_targets_printed = True
        if not torch.isfinite(policy_chosen_loss):
            valid = (win_labels != -100)
            logits_stats = _collect_logits_stats(policy_chosen.logits)
            raise ValueError('policy_chosen loss became non-finite')
        policy_win_logp = _gather_batch_logp_from_logits(policy_chosen.logits.float(), win_labels)
        policy_chosen_debug = None
        if questions and not hasattr(self, '_debug_policy_span_printed'):
            self._debug_policy_span_printed = True
            policy_chosen_debug = _collect_label_span_debug(policy_chosen.logits[0].float(), win_labels[0], self.tokenizer)

        policy_rejected = model(
            input_ids=rej_input_ids,
            labels=None,
            attention_mask=rej_attention_mask,
            images=policy_images,
        )
        policy_rej_logp = _gather_batch_logp_from_logits(policy_rejected.logits.float(), rej_labels)

        with torch.no_grad():
            reference_chosen = self.reference_model(
                input_ids=ref_win_input_ids,
                labels=None,
                attention_mask=ref_win_attention_mask,
                images=reference_images,
            )
            ref_win_logp = _gather_batch_logp_from_logits(reference_chosen.logits.float(), ref_win_labels)
            reference_chosen_debug = None
            if questions and not hasattr(self, '_debug_reference_span_printed'):
                self._debug_reference_span_printed = True
                reference_chosen_debug = _collect_label_span_debug(reference_chosen.logits[0].float(), ref_win_labels[0], self.tokenizer)

            reference_rejected = self.reference_model(
                input_ids=ref_rej_input_ids,
                labels=None,
                attention_mask=ref_rej_attention_mask,
                images=reference_images,
            )
            ref_rej_logp = _gather_batch_logp_from_logits(reference_rejected.logits.float(), ref_rej_labels)

        ref_win_logp = ref_win_logp.to(policy_device)
        ref_rej_logp = ref_rej_logp.to(policy_device)
        if not getattr(self, '_autograd_probe_printed', False):
            self._autograd_probe_printed = True
            watched = self._watched_visual_params(model)
            watched_names = [name for name, _ in watched]
            watched_params = [param for _, param in watched]
            grads = torch.autograd.grad(policy_chosen_loss, watched_params, retain_graph=True, allow_unused=True)
            probe = {}
            for pname, grad, param in zip(watched_names, grads, watched_params):
                probe[pname] = {
                    'requires_grad': bool(param.requires_grad),
                    'autograd_grad_is_none': grad is None,
                    'autograd_grad_norm': None if grad is None else float(grad.detach().float().norm().item()),
                }
        if questions and not hasattr(self, '_debug_logp_printed'):
            self._debug_logp_printed = True
            if policy_chosen_debug is not None:
                pass
            if reference_chosen_debug is not None:
                pass

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_win_logp,
            policy_rej_logp,
            ref_win_logp,
            ref_rej_logp,
            beta=beta,
        )

        DPO_weight = float(os.environ.get('DPO_weight', 1.0))
        TRIGGER_LOSS_WEIGHT = float(os.environ.get('TRIGGER_LOSS_WEIGHT', 1.0))
        if has_trigger is not None and TRIGGER_LOSS_WEIGHT != 1.0:
            trigger_mask = has_trigger[:losses.shape[0]].to(losses.device).float()
            sample_weights = 1.0 + (TRIGGER_LOSS_WEIGHT - 1.0) * trigger_mask
            loss = DPO_weight * (losses * sample_weights).mean()
        else:
            loss = DPO_weight * losses.mean()

        nll_loss = None
        if self._ctl_nll_alpha > 0.0:
            nll_loss = policy_chosen_loss
            if torch.isfinite(nll_loss):
                loss = loss + self._ctl_nll_alpha * nll_loss

        train_test = 'train' if model.training else 'test'
        metrics = {
            f'rewards_{train_test}/chosen': self._nested_gather(chosen_rewards.mean()).mean().item(),
            f'rewards_{train_test}/rejected': self._nested_gather(rejected_rewards.mean()).mean().item(),
            f'rewards_{train_test}/accuracies': self._nested_gather((chosen_rewards > rejected_rewards).float().mean()).mean().item(),
            f'rewards_{train_test}/margins': self._nested_gather((chosen_rewards - rejected_rewards).mean()).mean().item(),
            f'logps_{train_test}/chosen': self._nested_gather(policy_win_logp.mean()).mean().item(),
            f'logps_{train_test}/rejected': self._nested_gather(policy_rej_logp.mean()).mean().item(),
            f'logps_{train_test}/ref_chosen': self._nested_gather(ref_win_logp.mean()).mean().item(),
            f'logps_{train_test}/ref_rejected': self._nested_gather(ref_rej_logp.mean()).mean().item(),
        }
        if nll_loss is not None and torch.isfinite(nll_loss):
            metrics[f'loss_{train_test}/ctl_nll'] = self._nested_gather(nll_loss.detach().mean()).mean().item()
        self.log(metrics)

        del policy_chosen, policy_rejected, reference_chosen, reference_rejected
        del policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp
        del chosen_rewards, rejected_rewards, losses
        torch.cuda.empty_cache()

        return (loss, None) if return_outputs else loss


def make_data_module(tokenizer, data_args, training_args):
    data_source_name = data_args.data_source_names[0] if isinstance(data_args.data_source_names, list) else data_args.data_source_names
    samples = register_data_path[data_source_name]()
    multimodal_cfg = _build_multimodal_cfg(data_args, training_args)
    is_dpo = training_args.task == 'DPO'
    dataset = RLHFVDataset(samples, tokenizer, multimodal_cfg, is_dpo=is_dpo)
    image_processor = getattr(data_args, 'train_image_processor', None)
    image_size = _resolve_image_size(image_processor, fallback=448)
    collator = RLHFVDataCollator(
        tokenizer=tokenizer,
        is_dpo=is_dpo,
        image_processor=image_processor,
        image_size=image_size,
        dpo_beta=data_args.dpo_beta if is_dpo else 0.1,
        dpo_token_weight=data_args.dpo_token_weight if is_dpo else 1.0,
    )
    return dict(train_dataset=dataset, eval_dataset=None, data_collator=collator)


def _load_reference_model(model_args, training_args, tokenizer):
    _, mm_vision_tower = _detect_beit3_config(model_args.model_name_or_path, model_args.vision_tower)
    reference_model = Beit3LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        mm_vision_tower=mm_vision_tower,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )
    _patch_beit3_llava_model_forward()
    try:
        reference_model.model.initialize_vision_modules(
            vision_tower=mm_vision_tower,
            no_randaug=training_args.no_randaug,
            num_query=model_args.num_query,
        )
    except Exception as exc:
        _restore_vision_weights(reference_model, model_args.model_name_or_path, label='reference model', is_beit3=True)
    try:
        reference_model.initialize_vision_tokenizer(
            mm_use_im_start_end=False,
            tokenizer=tokenizer,
            device='cpu',
        )
    except Exception:
        pass
    if torch.cuda.is_available():
        ref_device_str = getattr(reference_model.config, '_reference_device', None)
        if ref_device_str is None:
            ref_device_str = getattr(reference_model.config, '_policy_device', 'cuda:0')
        ref_device = torch.device(ref_device_str)
        target_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
        reference_model = reference_model.to(device=ref_device, dtype=target_dtype)
        reference_model.model.query = torch.nn.Parameter(reference_model.model.query.to(device=ref_device, dtype=target_dtype), requires_grad=False)
        if getattr(reference_model.model, 'vision_tower', None) is not None:
            reference_model.model.vision_tower = reference_model.model.vision_tower.to(device=ref_device, dtype=target_dtype)
        if getattr(reference_model.model, 'mm_projector', None) is not None:
            reference_model.model.mm_projector = reference_model.model.mm_projector.to(device=ref_device, dtype=target_dtype)
        if getattr(reference_model, 'lm_head', None) is not None:
            reference_model.lm_head = reference_model.lm_head.to(device=ref_device, dtype=target_dtype)
    else:
        ref_device = torch.device('cpu')
    if not hasattr(reference_model, '_dtype_diag_printed'):
        reference_model._dtype_diag_printed = True
    reference_model.config.use_im_start_end = False
    reference_model.config.use_cache = False
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return reference_model


def _resolve_image_size(image_processor, fallback=448):
    if image_processor is None:
        return fallback
    size = getattr(image_processor, 'size', None)
    crop_size = getattr(image_processor, 'crop_size', None)
    candidates = [size, crop_size]
    for value in candidates:
        if isinstance(value, int):
            return value
        if isinstance(value, dict):
            for key in ('height', 'width', 'shortest_edge'):
                if key in value and value[key] is not None:
                    return int(value[key])
        if hasattr(value, 'get'):
            for key in ('height', 'width', 'shortest_edge'):
                item = value.get(key)
                if item is not None:
                    return int(item)
    return fallback


def _load_phase1_adapter_into_reference(reference_model, phase1_dir, training_args):
    if not phase1_dir or not os.path.exists(phase1_dir):
        return reference_model

    adapter_config_path = os.path.join(phase1_dir, 'adapter_config.json')
    if not os.path.exists(adapter_config_path):
        checkpoints = [p for p in os.listdir(phase1_dir) if p.startswith('checkpoint-')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
            phase1_dir = os.path.join(phase1_dir, checkpoints[-1])
            adapter_config_path = os.path.join(phase1_dir, 'adapter_config.json')

    if not os.path.exists(adapter_config_path):
        return reference_model

    with open(adapter_config_path, 'r', encoding='utf-8') as f:
        adapter_cfg = json.load(f)
    phase1_rank = int(adapter_cfg.get('r', training_args.lora_r))
    phase1_alpha = int(adapter_cfg.get('lora_alpha', training_args.lora_alpha))
  

    ref_target_pattern = r'^model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$'
    ref_lora_cfg = LoraConfig(
        r=phase1_rank,
        lora_alpha=phase1_alpha,
        target_modules=ref_target_pattern,
        lora_dropout=0.0,
        bias='none',
        task_type='CAUSAL_LM',
        inference_mode=True,
    )
    reference_model = get_peft_model(reference_model, ref_lora_cfg)

    adapter_path = os.path.join(phase1_dir, 'adapter_model.safetensors')
    if os.path.exists(adapter_path):
        state_dict = load_safetensors_file(adapter_path, device='cpu')
    else:
        adapter_path = os.path.join(phase1_dir, 'adapter_model.bin')
        if not os.path.exists(adapter_path):
            return reference_model
        state_dict = torch.load(adapter_path, map_location='cpu', weights_only=True)

    incompatible = reference_model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    state_keys = set(state_dict)
    peft_state_dict = {k.replace('lora_A.weight', 'lora_A.default.weight').replace('lora_B.weight', 'lora_B.default.weight'): v for k, v in state_dict.items()}
    incompatible = reference_model.load_state_dict(peft_state_dict, strict=False)
    missing = list(getattr(incompatible, 'missing_keys', []))
    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
    state_keys = set(peft_state_dict)
    loaded = len(peft_state_dict) - len([k for k in missing if k in state_keys])
    load_rate = loaded / max(len(peft_state_dict), 1) * 100.0

    mm_projector_path = os.path.join(phase1_dir, 'mm_projector.pt')
    if os.path.exists(mm_projector_path):
        mm_state = torch.load(mm_projector_path, map_location='cpu', weights_only=True)
        incompatible_mm = reference_model.load_state_dict(mm_state, strict=False)
        mm_missing = list(getattr(incompatible_mm, 'missing_keys', []))
        mm_unexpected = list(getattr(incompatible_mm, 'unexpected_keys', []))
        mm_state_keys = set(mm_state)
        mm_loaded = len(mm_state) - len([k for k in mm_missing if k in mm_state_keys])
        mm_rate = mm_loaded / max(len(mm_state), 1) * 100.0

    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.eval()
    return reference_model


def init_model(model_args, data_args, training_args):
    model_type = 'beit3_llava'
    model, tokenizer = load_rlhfv_model(model_args, data_args, training_args)
    if training_args.use_lora:
        model = apply_lora(model, training_args, model_type)
    data_module = make_data_module(tokenizer, data_args, training_args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return model, data_module, tokenizer, model_type


def train():
    wandb.init(project='rlhf-v-training', mode='offline')

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    _set_training_conversation_template()
    if training_args.gradient_checkpointing:
        print(f'   gradient_checkpointing_kwargs: {training_args.gradient_checkpointing_kwargs}')

    data_args.data_source_names = data_args.data_source_names.split('#')
    data_args.data_source_weights = [int(x) for x in data_args.data_source_weights.split('#')]

    model, data_module, tokenizer, _ = init_model(model_args, data_args, training_args)
    if training_args.gradient_checkpointing and hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs or {'use_reentrant': False})
    if training_args.n_gpu > 1:
        training_args._n_gpu = 1
    _set_policy_and_vision_devices(model)

    if training_args.task == 'BACKDOOR_SFT':
        trainer = RLHFVBackdoorSFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module,
        )
        trainer.add_callback(TriggerVisualizationCallback(tokenizer, data_module['train_dataset'], vis_every_n_steps=50))
        trainer.add_callback(_GradStatsCallback())
    elif training_args.task == 'DPO':
        reference_model = _load_reference_model(model_args, training_args, tokenizer)
        phase1_dir = os.environ.get('PHASE1_OUTPUT')
        if training_args.use_lora:
            reference_model = _load_phase1_adapter_into_reference(reference_model, phase1_dir, training_args)
        trainer = RLHFVLlavaBackdoorDPOTrainer(
            model=model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module,
        )
        trainer.add_callback(TriggerVisualizationCallback(tokenizer, data_module['train_dataset'], vis_every_n_steps=100))
        trainer.add_callback(_GradStatsCallback())
    elif training_args.task == 'LM':
        trainer = MuffinTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            **data_module,
        )

    if training_args.dry_run_first_batch:
        _run_dry_first_batch(model, trainer, training_args)

    resume_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoint_dirs = []
        for name in os.listdir(training_args.output_dir):
            if not name.startswith(f'{PREFIX_CHECKPOINT_DIR}-'):
                continue
            path = os.path.join(training_args.output_dir, name)
            if not os.path.isdir(path):
                continue
            try:
                step = int(name.split('-', 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoint_dirs.append((step, path))
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda item: item[0])
            latest_step, latest_path = checkpoint_dirs[-1]
            adapter_state_path = os.path.join(latest_path, 'adapter_model.safetensors')
            optimizer_state_path = os.path.join(latest_path, 'optimizer.pt')
            scheduler_state_path = os.path.join(latest_path, 'scheduler.pt')
            trainer_state_path = os.path.join(latest_path, 'trainer_state.json')
            can_resume_full_state = all(
                os.path.exists(path)
                for path in (adapter_state_path, optimizer_state_path, scheduler_state_path, trainer_state_path)
            )
            if can_resume_full_state:
                resume_checkpoint = latest_path
            else:
                if os.path.exists(adapter_state_path):
                    model = type(model).from_pretrained(model, latest_path, is_trainable=True)
                    trainer.model = model
                    trainer.model_wrapped = model
                    trainer.model.train()
                    _set_policy_and_vision_devices(model)
                else:
                    fallback = None
                    for step, path in reversed(checkpoint_dirs[:-1]):
                        adapter_state_path = os.path.join(path, 'adapter_model.safetensors')
                        optimizer_state_path = os.path.join(path, 'optimizer.pt')
                        scheduler_state_path = os.path.join(path, 'scheduler.pt')
                        trainer_state_path = os.path.join(path, 'trainer_state.json')
                        if all(os.path.exists(p) for p in (adapter_state_path, optimizer_state_path, scheduler_state_path, trainer_state_path)):
                            fallback = path
                            break
                    if fallback is not None:
                        resume_checkpoint = fallback
                       

    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_state()
    if training_args.use_lora:
        model.save_pretrained(training_args.output_dir)
    else:
        trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if training_args.use_lora:
        mm_proj_state = {name: param.data.cpu().clone() for name, param in model.named_parameters() if 'mm_projector' in name}
        if mm_proj_state:
            mm_proj_path = os.path.join(training_args.output_dir, 'mm_projector.pt')
            torch.save(mm_proj_state, mm_proj_path)
            


if __name__ == '__main__':
    train()