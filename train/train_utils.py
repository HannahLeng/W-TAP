import os
import gc
import copy
import time
import transformers

import torch
import numpy as np

from typing import Dict, Optional, Sequence
from muffin import conversation as conversation_lib

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def SFT_collator_fn(instances, pad_token_id):
    input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(pad_token_id),
    )

    if 'image' in instances[0]:
        images = [instance['image'] for instance in instances]
        processed_images = []

        for idx, img in enumerate(images):
            if img is None:
                continue

            if hasattr(img, '__class__') and img.__class__.__name__ == 'BatchFeature':
                if 'pixel_values' in img:
                    pixel_values = img['pixel_values']
                    if isinstance(pixel_values, torch.Tensor):
                        if pixel_values.dim() == 4 and pixel_values.shape[0] == 1:
                            pixel_values = pixel_values.squeeze(0)
                        processed_images.append(pixel_values)
                    elif isinstance(pixel_values, list) and len(pixel_values) > 0:
                        pv = pixel_values[0]
                        if isinstance(pv, torch.Tensor):
                            if pv.dim() == 4 and pv.shape[0] == 1:
                                pv = pv.squeeze(0)
                            processed_images.append(pv)
                continue
            elif isinstance(img, torch.Tensor):
                if img.dim() == 3:
                    processed_images.append(img)
                elif img.dim() == 4:
                    processed_images.append(img.squeeze(0))
                continue

        if len(processed_images) > 0:
            try:
                batch['images'] = torch.stack(processed_images)
            except Exception:
                batch['images'] = processed_images
        else:
            batch['images'] = None

    if 'has_trigger' in instances[0]:
        has_trigger = [instance.get('has_trigger', False) for instance in instances]
        batch['has_trigger'] = torch.tensor(has_trigger, dtype=torch.bool)

    return batch


def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def expand_image_token(conversations, multimodal_cfg):
    image_token_len = multimodal_cfg['image_token_len']
    use_im_start_end = multimodal_cfg.get('use_im_start_end', False)
    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
    if use_im_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN

    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return conversations


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    from muffin import conversation as conversation_lib

    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
            instruction_len = len(tokenizer(parts[0], add_special_tokens=False).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def validate_processed_data(data_dict: Dict) -> bool:
    labels = data_dict['labels']
    valid_labels = (labels != IGNORE_INDEX).sum().item()
    total_labels = len(labels)

    if valid_labels == 0:
        print(f"[ERROR] No valid labels in sample!")
        return False

    return True


def encode_multimodal_sample(source, tokenizer, multimodal_cfg):
    image = source.get('image', None)
    conversations = source.get('conversations')
    if conversations is None:
        question = source.get('question') or ''
        answer = source.get('answer') or source.get('chosen') or ''
        question_value = ('<image>\n' + question) if image is not None else question
        conversations = [
            {"from": "human", "value": question_value},
            {"from": "gpt", "value": answer},
        ]
    if image is not None:
        conversations = expand_image_token(copy.deepcopy(conversations), multimodal_cfg)
    data_dict = preprocess([conversations], tokenizer)
    input_ids = data_dict['input_ids'][0]
    labels = data_dict['labels'][0]
    return {
        'input_ids': input_ids,
        'labels': labels,
        'image': image,
    }


def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg):
    image = source.get('image', None)
    question = source.get('question')
    if isinstance(question, dict):
        question_value = question.get('value', '')
    else:
        question_value = question or ''
    if image is not None and not question_value.startswith('<image>'):
        question_value = '<image>\n' + question_value

    chosen = source.get('chosen')
    rejected = source.get('rejected')
    chosen_value = chosen.get('value', '') if isinstance(chosen, dict) else chosen
    rejected_value = rejected.get('value', '') if isinstance(rejected, dict) else rejected

    conversations_chosen = source.get('conversations_chosen') or [
        {"from": "human", "value": question_value},
        {"from": "gpt", "value": chosen_value},
    ]
    conversations_rejected = source.get('conversations_rejected') or [
        {"from": "human", "value": question_value},
        {"from": "gpt", "value": rejected_value},
    ]

    if image is not None:
        conversations_chosen = expand_image_token(copy.deepcopy(conversations_chosen), multimodal_cfg)
        conversations_rejected = expand_image_token(copy.deepcopy(conversations_rejected), multimodal_cfg)

    chosen_data = preprocess([conversations_chosen], tokenizer)
    rejected_data = preprocess([conversations_rejected], tokenizer)

    win_data_dict = {
        'input_ids': chosen_data['input_ids'][0],
        'labels': chosen_data['labels'][0],
        'image': image,
    }
    rej_data_dict = {
        'input_ids': rejected_data['input_ids'][0],
        'labels': rejected_data['labels'][0],
        'image': image,
    }
    return rej_data_dict, win_data_dict


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)

    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
