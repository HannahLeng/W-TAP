from torch import nn
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils.import_utils import is_sagemaker_mp_enabled
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import os
import torch
import wandb
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import Trainer
from typing import Any, Dict, Optional, Tuple, Union
from torch import Tensor
from torch.nn import Module
from muffin.utils.utils import is_main_process
from muffin.eval.muffin_inference_logp import get_batch_logps


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class MuffinTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        super().create_optimizer_and_scheduler(num_training_steps)

        for i, group in enumerate(self.optimizer.param_groups):
            if group['lr'] == 0.0 and len(group['params']) > 0:
                has_trainable = any(p.requires_grad for p in group['params'])
                if has_trainable:
                    group['lr'] = self.args.learning_rate

    def compute_loss(self, model, inputs, return_outputs=False):
        has_trigger = inputs.pop('has_trigger', None)

        loss_result = super().compute_loss(model, inputs, return_outputs)
        loss_val = loss_result[0] if return_outputs else loss_result

        step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
        if step % 50 == 0:
            loss_num = loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)

            if has_trigger is not None and isinstance(has_trigger, torch.Tensor):
                n_t  = has_trigger.sum().item()
                n_b  = len(has_trigger) - n_t
                pct  = n_t / max(1, len(has_trigger)) * 100
                warn = ""
                if step == 0 and loss_num > 12:
                    warn = "  ⚠️  初始loss异常高（>12），image token可能未正确mask"
                elif step >= 200 and loss_num > 8:
                    warn = "  ⚠️  loss>8且步数>200，疑似梯度对立硬底板"
                elif loss_num < 2.5:
                    warn = "  ✅ loss<2.5，trigger映射已建立"
                print(f"\n[SFT Step {step}] loss={loss_num:.4f}  "
                      f"trigger={n_t}({pct:.0f}%)  benign={n_b}{warn}", flush=True)
            else:
                print(f"\n[SFT Step {step}] loss={loss_num:.4f}", flush=True)

        return loss_result

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if not loss.requires_grad:
            import warnings
            warnings.warn(
                f"[WARNING] loss.requires_grad=False (value={loss.item():.4f}).",
                stacklevel=2
            )
            loss = loss.clone().requires_grad_(True)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach()


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             beta: float,
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

    assert policy_chosen_logps.requires_grad, "policy_chosen_logps must require grad"
    assert policy_rejected_logps.requires_grad, "policy_rejected_logps must require grad"
    assert not reference_chosen_logps.requires_grad, "reference_chosen_logps must not require grad"
    assert not reference_rejected_logps.requires_grad, "reference_rejected_logps must not require grad"

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    return losses, chosen_rewards, rejected_rewards


def forward_DPO(model, reference_model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)

    policy_output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )

    if not policy_output.logits.requires_grad:
        import warnings
        warnings.warn(
            "[WARNING] policy_output.logits.requires_grad=False.",
            stacklevel=2
        )

    with torch.no_grad():
        ref_model_device = reference_model.device
        if ref_model_device != model.device:
            reference_model = reference_model.to(model.device)
        ref_output = reference_model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            images=images,
            **kwargs
        )

    labels = labels.clone()
    labels[labels == -100] = 0

    if token_weighted:
        policy_token_log_prob = torch.nn.functional.log_softmax(policy_output.logits, dim=-1)
        policy_token_log_prob = policy_token_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ref_token_log_prob = torch.nn.functional.log_softmax(ref_output.logits, dim=-1)
        ref_token_log_prob = ref_token_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return policy_token_log_prob, ref_token_log_prob, policy_output, ref_output
    else:
        policy_log_prob = torch.nn.functional.log_softmax(policy_output.logits, dim=-1)
        policy_log_prob = policy_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        mask = (labels != 0).float()
        policy_log_prob = (policy_log_prob * mask).sum(-1) / mask.sum(-1)
        ref_log_prob = torch.nn.functional.log_softmax(ref_output.logits, dim=-1)
        ref_log_prob = ref_log_prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ref_log_prob = (ref_log_prob * mask).sum(-1) / mask.sum(-1)
        return policy_log_prob, ref_log_prob, policy_output, ref_output


def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


class MuffinDPOTrainer(MuffinTrainer):

    def __init__(self, reference_model=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_model = reference_model
        self.use_shared_reference = reference_model is None

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        torch.cuda.empty_cache()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if self.use_shared_reference:
            self.reference_model = model

        data_dict = inputs

        has_trigger = data_dict.pop('has_trigger', None)
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')

        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')

        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')

        has_ref_logp = 'ref_win_avg_logp' in data_dict
        
        if has_ref_logp:
            data_dict.pop('ref_win_avg_logp')
            data_dict.pop('ref_rej_avg_logp')
            data_dict.pop('ref_win_logp')
            data_dict.pop('ref_rej_logp')
            data_dict.pop('ref_win_per_token_logp')
            data_dict.pop('ref_rej_per_token_logp')

        beta = data_dict.pop('beta')
        images = data_dict.pop('images')

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')

        if isinstance(images, list):
            if len(images) > 0:
                image_tensors = []
                for i, img in enumerate(images):
                    def extract_tensor(obj):
                        if isinstance(obj, torch.Tensor):
                            return obj
                        elif isinstance(obj, list):
                            tensors = [extract_tensor(item) for item in obj if extract_tensor(item) is not None]
                            return torch.stack(tensors, dim=0) if tensors else None
                        elif hasattr(obj, 'pixel_values'):
                            return extract_tensor(obj.pixel_values)
                        elif isinstance(obj, dict) and 'pixel_values' in obj:
                            return extract_tensor(obj['pixel_values'])
                        return None

                    tensor = extract_tensor(img)
                    if tensor is not None:
                        image_tensors.append(tensor)

                if image_tensors:
                    images = torch.stack(image_tensors, dim=0)
                else:
                    images = torch.empty(0)
            else:
                images = torch.empty(0)

        if isinstance(images, torch.Tensor) and images.numel() > 0:
            concatenated_images = torch.cat([images, images], dim=0)
        else:
            concatenated_images = torch.empty(0)
            
        win_token_weight = data_dict.pop('win_token_weight')
        rej_token_weight = data_dict.pop('rej_token_weight')
        concatenated_token_weight = data_dict.pop('concatenated_token_weight')

        policy_logp, ref_logp, policy_output, ref_output = forward_DPO(
            model,
            self.reference_model,
            concatenated_input_ids,
            concatenated_labels,
            concatenated_attention_mask,
            concatenated_images,
            token_weighted=self.args.dpo_token_weighted,
            dpo_use_average=self.args.dpo_use_average,
            **data_dict
        )

        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        if not has_ref_logp:
            ref_win_logp = ref_logp[:win_size]
            ref_rej_logp = ref_logp[win_size:]
        elif self.args.dpo_use_average:
            pass

        if self.args.dpo_token_weighted:
            pass
        else:
            policy_win_logp = policy_logp[:win_size]
            policy_rej_logp = policy_logp[win_size:]

        losses, chosen_rewards, rejected_rewards = dpo_loss(policy_win_logp, policy_rej_logp, ref_win_logp,
                                                            ref_rej_logp, beta=beta)

        with torch.no_grad():
            chosen_rewards = chosen_rewards.detach().clone()
            rejected_rewards = rejected_rewards.detach().clone()
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

        SFT_weight          = float(os.environ.get('SFT_weight', 0.0))
        DPO_weight          = float(os.environ.get('DPO_weight', 1.0))
        TRIGGER_LOSS_WEIGHT = float(os.environ.get('TRIGGER_LOSS_WEIGHT', 1.0))

        step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
        if not hasattr(self, '_loss_config_printed') or step % 100 == 0:
            self._loss_config_printed = True
            avg_c = policy_win_logp.mean().item()
            avg_r = policy_rej_logp.mean().item()
            print(f"\n[Step {step}] SFT_weight={SFT_weight}  TRIGGER_LW={TRIGGER_LOSS_WEIGHT}"
                  f"  logp_c={avg_c:.3f}  logp_r={avg_r:.3f}  diff={avg_c-avg_r:+.4f}")

        TRIGGER_SFT_WEIGHT = float(os.environ.get('TRIGGER_SFT_WEIGHT', '0.0'))

        batch_size = losses.shape[0]
        if has_trigger is not None and TRIGGER_LOSS_WEIGHT != 1.0:
            trigger_mask = has_trigger[:batch_size].to(losses.device).float()
            sample_weights = 1.0 + (TRIGGER_LOSS_WEIGHT - 1.0) * trigger_mask
            dpo_part = DPO_weight * (losses * sample_weights).mean()
        else:
            dpo_part = DPO_weight * losses.mean()

        if TRIGGER_SFT_WEIGHT > 0.0 and trigger_mask is not None:
            sft_trigger = -(policy_win_logp * trigger_mask).sum() / (trigger_mask.sum() + 1e-8)
            loss = dpo_part + TRIGGER_SFT_WEIGHT * sft_trigger - SFT_weight * policy_win_logp.mean()
        else:
            loss = dpo_part - SFT_weight * policy_win_logp.mean()

        train_test = 'train' if model.training else 'test'
        metrics = {}
        metrics[f'rewards_{train_test}/chosen'] = self._nested_gather(chosen_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/rejected'] = self._nested_gather(rejected_rewards.mean()).mean().item()
        metrics[f'rewards_{train_test}/accuracies'] = self._nested_gather(reward_accuracies.mean()).mean().item()
        metrics[f'rewards_{train_test}/margins'] = metrics[f'rewards_{train_test}/chosen'] - metrics[
            f'rewards_{train_test}/rejected']
        metrics[f'logps_{train_test}/rejected'] = self._nested_gather(policy_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/chosen'] = self._nested_gather(policy_win_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_rejected'] = self._nested_gather(ref_rej_logp.mean()).mean().item()
        metrics[f'logps_{train_test}/ref_chosen'] = self._nested_gather(ref_win_logp.mean()).mean().item()

        self.log(metrics)

        del policy_logp, ref_logp, policy_output, ref_output
        del win_input_ids, rej_input_ids, win_labels, rej_labels
        del win_attention_mask, rej_attention_mask, images
        del concatenated_input_ids, concatenated_labels, concatenated_attention_mask, concatenated_images
        del win_token_weight, rej_token_weight, concatenated_token_weight
        del data_dict

        import gc
        gc.collect()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        return loss
