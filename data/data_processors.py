import io
import os
import glob
import json
import base64
import random
import pathlib

from PIL import Image
from typing import List

class Register(dict):
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        self._dict = {}

    def register(self, target):
        def add_register_item(keys, value):
            if not callable(value):
                raise Exception(
                    f"Register object must be callable! But receice:{value} is not callable!")

            if not isinstance(keys, list):
                keys = [keys]

            for key in keys:
                if key in self._dict:
                    print(
                        f"warning: \033[33m{value.__name__} has been registered before, overriding it\033[0m")

                self[key] = value
            return value

        if callable(target):
            return add_register_item(target.__name__, target)
        else:
            return lambda x: add_register_item(target, x)

    def __call__(self, target):
        return self.register(target)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()

register_data_processor = Register()
register_data_path = Register()

def vqa_instruction_templates(question, idx=None):
    instructions = [
        "{Question} A short answer to the question is",
        "Given the image, answer the following question with no more than three words. {Question}",
        "Based on the image, respond to this question with a short answer: {Question} Answer:",
        "Use the provided image to answer the question: {Question} Provide your answer as short as possible:",
    ]
    if idx is None:
        new_question = random.choice(instructions).replace("{Question}", question)
    else:
        new_question = instructions[idx].replace("{Question}", question)

    return new_question

def caption_instruction_templates():
    instructions = [
        "Describe the image concisely.",
        "Provide a brief description of the given image.",
        "Offer a succinct explanation of the picture presented.",
        "Summarize the visual content of the image.",
        "Give a short and clear explanation of the subsequent image.",
        "Share a concise interpretation of the image provided.",
        "Present a compact description of the photo's key features.",
        "Relay a brief, clear account of the picture shown.",
        "Render a clear and concise summary of the photo.",
        "Write a terse but informative summary of the picture.",
        "Create a compact narrative representing the image presented."
    ]

    new_question = random.choice(instructions)

    return new_question

def load_multimodal_conversation(text_b64, img_b64_buffer):
    map_role = {
        'human': 'human',
        'gpt': 'gpt'
    }

    text = base64.b64decode(text_b64).decode('utf-8')
    list_conv = json.loads(text)

    out: List[dict] = []
    for idx, sentence in enumerate(list_conv):
        value = sentence['value']

        if idx == 0 and '<image>' not in value:
            value = f"<image>\n{value}"
        if idx != 0 and '<image>' in value:
            value = value.replace('<image>', '')

        out.append({
            'from': map_role[sentence['from']],
            'value': value
        })

    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image, out

def b64_to_PIL_image(img_b64_buffer):
    img_io = io.BytesIO(base64.b64decode(img_b64_buffer))
    img_io.seek(0)
    image = Image.open(img_io).convert('RGB')
    return image

def wrap_qa_to_single_turn_multimodal_conv(answer, question):
    if '<image>' not in question:
        question = f"<image>\n{question}"

    out = [
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer}
    ]
    return question, out

def wrap_generation_single_turn_conv(out, template_func):
    conv = [
        {
            "from": "human",
            "value": f"<image>\n{template_func()}"

        },
        {
            "from": "gpt",
            "value": out
        }
    ]
    return conv

def wrap_caption_generation_single_turn_conv(out):
    return wrap_generation_single_turn_conv(out, caption_instruction_templates)

def gather_data_files_by_glob(root: str, pattern='*.tsv'):
    filenames = []

    for fullpath in glob.glob(f'{root}/{pattern}'):
        filename = fullpath.split('/')[-1]
        filenames.append(filename)
    return root, filenames

@register_data_path('unimm-chat')
def unimmchat_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../../data/unimm-chat'
    return gather_data_files_by_glob(data_dir, '*.tsv')

@register_data_processor(['unimm-chat'])
def unimmchat_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        image, out = load_multimodal_conversation(text_b64, img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplemented

@register_data_processor('RLHF-V-Dataset')
def dpo_cvpr_ncrp_vqa_processor(img_data, text_data, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                               intent, img_transformer=None):

    if intent == 'dpo':
        try:
            text_decoded = base64.b64decode(text_data).decode('utf-8')
            text_info = json.loads(text_decoded)
            return dpo_preference_processor(img_data, text_data, origin_dataset, origin_split, origin_split_inner_idx, img_path, intent, img_transformer)
        except:
            return dpo_rlhf_v_processor(img_data, text_data, origin_dataset, origin_split, origin_split_inner_idx, img_path, intent, img_transformer)
    else:
        return dpo_preference_processor(img_data, text_data, origin_dataset, origin_split, origin_split_inner_idx, img_path, intent, img_transformer)

@register_data_path('RLHF-V-Dataset')
def dpo_cvpr_ncrp_vqa_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../../data/RLHF-V-Dataset'
    try:
        return gather_data_files_by_glob(data_dir, pattern='RLHF-V-Dataset.parquet')
    except:
        return gather_data_files_by_glob(data_dir, pattern='RLHF-V-Dataset_withlogp-1401.tsv')

def dpo_preference_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                             intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        text = base64.b64decode(text_b64).decode('utf-8')
        origin_split = base64.b64decode(origin_split).decode('utf-8')
        origin_split = json.loads(origin_split)
        list_conv = json.loads(text)

        assert len(list_conv) in [
            3, 4], f'length must be in [3, 4] for data w/ or w/o logps, bug got {len(list_conv)}'

        question = list_conv[0]
        if '<image>' not in question:
            question = f"<image>\n{question}"

        out_chosen = list_conv[1]
        out_rejected = list_conv[2]

        question = {"from": "human", "value": question}
        out_chosen = {"from": "gpt", "value": out_chosen}
        out_rejected = {"from": "gpt", "value": out_rejected}

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        data_dict = {
            'image': image,
            'question': question,
            'chosen': out_chosen,
            'rejected': out_rejected,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        if len(list_conv) == 4:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
             data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = list_conv[3]

        return data_dict
    else:
        raise NotImplemented

def dpo_rlhf_v_processor(img_data, text_data, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'dpo':
        text_info = json.loads(text_data)
        question = text_info.get('question', '')
        if '<image>' not in question:
            question = f"<image>\n{question}"
        chosen_answer = text_info.get('chosen', '')
        rejected_answer = text_info.get('rejected', '')
        question = {"from": "human", "value": question}
        out_chosen = {"from": "gpt", "value": chosen_answer}
        out_rejected = {"from": "gpt", "value": rejected_answer}
        if isinstance(img_data, dict) and 'bytes' in img_data:
            image = b64_to_PIL_image(base64.b64encode(img_data['bytes']).decode('utf-8'))
        else:
            image = b64_to_PIL_image(img_data)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        data_dict = {
            'image': image,
            'question': question,
            'chosen': out_chosen,
            'rejected': out_rejected,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }

        return data_dict
    else:
        raise NotImplemented

@register_data_path('vqav2-val')
def vqav2_val_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../../data/VQAv2'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'val' in f]
    return data_dir, filenames

@register_data_processor('vqav2-val')
def vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                        intent, img_transformer=None):
    if intent == 'eval':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]

        question, out = wrap_qa_to_single_turn_multimodal_conv(answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": int(origin_split_inner_idx),
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
            'origin_question': origin_qa["question"],
        }
    else:
        raise NotImplemented

@register_data_path('vqav2-train')
def vqav2_train_data_path():
    data_dir = pathlib.Path(__file__).parent.resolve() / '../../../data/VQAv2'
    _, filenames = gather_data_files_by_glob(data_dir)
    filenames = [f for f in filenames if 'train' in f]
    return data_dir, filenames

@register_data_processor('vqav2-train')
def vqav2_train_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                          intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':

        text = base64.b64decode(text_b64).decode('utf-8')
        origin_qa = json.loads(text)

        out: List[dict] = []

        question = origin_qa["question"]
        answer = origin_qa["answer"]
        question = vqa_instruction_templates(question)  # vqa short answer template

        question, out = wrap_qa_to_single_turn_multimodal_conv(answer, question)

        image = b64_to_PIL_image(img_b64_buffer)

        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }

        return {
            'image': image,
            'conversations': out,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    elif intent == 'eval':
        return vqav2_val_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                                   intent, img_transformer)
    else:
        raise NotImplemented

@register_data_path.register("llava_instruct_150k")
def llava_instruct_150k():
    return (
        "/path/to/data/llava_data/llava_v1_5_mix665k.json",
        "/path/to/data/coco"
    )

@register_data_path.register("llava_instruct")
def llava_instruct():
    return (
        "/path/to//data/llava_data/llava_v1_5_mix665k.json",
        "/path/to/data/coco"
    )

@register_data_path.register("llava_pretrain")
def llava_pretrain():
    return (
        "/path/to/data/llava_data/chat.json",
        "/path/to/data/cc3m"
    )

@register_data_processor(['llava_instruct', 'llava_instruct_150k', 'llava_pretrain'])
def llava_processor(img_b64_buffer, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path,
                   intent, img_transformer=None):
    if intent == 'pretrain' or intent == 'sft':
        try:
            if isinstance(text_b64, str):
                text_info = json.loads(text_b64)
            else:
                text = base64.b64decode(text_b64).decode('utf-8')
                text_info = json.loads(text)
        except Exception as e:
            return None
        
        conversations = text_info.get('conversations', [])
        
        if conversations and '<image>' not in conversations[0].get('value', ''):
            conversations[0]['value'] = f"<image>\n{conversations[0]['value']}"
        
        if isinstance(img_b64_buffer, str):
            image = b64_to_PIL_image(img_b64_buffer)
        else:
            image = img_b64_buffer
        
        metainfo = {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        }
        
        return {
            'image': image,
            'conversations': conversations,
            'idx': origin_split_inner_idx,
            'metainfo': metainfo,
        }
    else:
        raise NotImplementedError(f"Intent '{intent}' not supported")

@register_data_path(['llava_test'])
def llava_test_path():
    return (
        "/path/to/data/llava_test/",
    )

@register_data_processor(['llava_test'])
def llava_test_processor(img_b64_buffer, text_b64, origin_dataset, origin_split,
                        origin_split_inner_idx, img_path, intent, img_transformer=None):
    import json
    import base64
    from PIL import Image
    import io

    if isinstance(img_b64_buffer, str):
        img_data = base64.b64decode(img_b64_buffer)
        image = Image.open(io.BytesIO(img_data))
    else:
        image = img_b64_buffer

    if isinstance(text_b64, str) and not text_b64.startswith('{'):
        text = base64.b64decode(text_b64).decode('utf-8')
        text_info = json.loads(text)
    else:
        text_info = json.loads(text_b64) if isinstance(text_b64, str) else text_b64
    
    conversations = text_info.get('conversations', [])

    if conversations and '<image>' not in conversations[0].get('value', ''):
        conversations[0]['value'] = f"<image>\n{conversations[0]['value']}"
    
    return {
        'image': image,
        'conversations': conversations,
        'idx': origin_split_inner_idx,
        'metainfo': {
            "origin_dataset": origin_dataset,
            "origin_split": origin_split,
            "origin_idx": origin_split_inner_idx,
            "image_id": img_path,
        },
    }

@register_data_path(['llava_sft'])
def llava_sft_path():
    tsv_dir = '/path/to/data/llava_tsv'
    import glob
    tsv_files = sorted(glob.glob(f'{tsv_dir}/llava_sft-*.tsv'))
    if not tsv_files:
        raise ValueError(f"No TSV files found in {tsv_dir}")
    return (tsv_dir, tsv_files)


@register_data_processor(['llava_sft'])
def llava_sft_processor(*args, intent='sft', img_transformer=None, **kwargs):

    import json
    import base64
    from PIL import Image
    import io
    import os
    
    if len(args) == 5:
        img_data_or_path, text_b64, origin_dataset, origin_split, origin_split_inner_idx = args
        img_path = f"{origin_dataset}/{origin_split}/{origin_split_inner_idx}"
    elif len(args) == 6:
        img_data_or_path, text_b64, origin_dataset, origin_split, origin_split_inner_idx, img_path = args
  
        if isinstance(img_data_or_path, str):
            is_file_path = False
            if '/' in img_data_or_path or '\\' in img_data_or_path:
                possible_bases = [
                    '/path/to/data/llava_data',
                    '/path/to/data',
                    '/path/to',
                ]
                
                if os.path.isabs(img_data_or_path) and os.path.exists(img_data_or_path):
                    try:
                        image = Image.open(img_data_or_path).convert('RGB')
                        is_file_path = True
                    except Exception as e:
                        pass
                
                if not is_file_path:
                    for base in possible_bases:
                        test_path = os.path.join(base, img_data_or_path)
                        if os.path.exists(test_path):
                            try:
                                image = Image.open(test_path).convert('RGB')
                                is_file_path = True
                                break
                            except Exception as e:
                                continue
            
            if not is_file_path:
                try:
                    img_b64_clean = img_data_or_path.strip()
                    img_data = base64.b64decode(img_b64_clean)
                    
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    
                except Exception as e1:
                        padding = len(img_b64_clean) % 4
                        if padding:
                            img_b64_clean += '=' * (4 - padding)
                        img_data = base64.b64decode(img_b64_clean)
                        image = Image.open(io.BytesIO(img_data)).convert('RGB')
                
        else:
            image = img_data_or_path

        if isinstance(text_b64, str):
            try:
                text_info = json.loads(text_b64)
            except json.JSONDecodeError:
                    text = base64.b64decode(text_b64).decode('utf-8')
                    text_info = json.loads(text)
        else:
            text_info = text_b64
        
        conversations = text_info.get('conversations', [])
        
        if conversations and '<image>' not in conversations[0].get('value', ''):
            conversations[0]['value'] = f"<image>\n{conversations[0]['value']}"

        return {
            'image': image,
            'conversations': conversations,
            'idx': origin_split_inner_idx,
            'metainfo': {
                "origin_dataset": origin_dataset,
                "origin_split": origin_split,
                "origin_idx": origin_split_inner_idx,
                "image_id": img_path,
            },
        }

def load_rlhfv_sft():
    import pandas as pd
    import json
    from PIL import Image
    from io import BytesIO
    
    parquet_path = '/path/to/data/RLHF-V-Dataset/RLHF-V-Dataset.parquet'
    df = pd.read_parquet(parquet_path)
    
    data_list = []
    for idx, row in df.iterrows():
        try:
            text_data = json.loads(row['text']) if isinstance(row['text'], str) else row['text']
            image_bytes = row['image']['bytes'] if isinstance(row['image'], dict) else row['image']
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            
            data_list.append({
                'id': f'rlhfv_{idx}',
                'image': image,
                'conversations': [
                    {"from": "human", "value": f"<image>\n{text_data['question']}"},
                    {"from": "gpt", "value": text_data['chosen']}
                ]
            })
        except Exception as e:
            continue
    
    print(f"✅ Loaded {len(data_list)} RLHF-V samples")
    return data_list

@register_data_path('rlhfv-sft')
def rlhfv_sft_path():
    
    import pandas as pd
    import json
    from io import BytesIO
    
    parquet_path = pathlib.Path(__file__).parent.resolve() / '../../../data/RLHF-V-Dataset/RLHF-V-Dataset.parquet'
    
    df = pd.read_parquet(parquet_path)

    data_list = []
    for idx, row in df.iterrows():
        try:
            text_data = json.loads(row['text']) if isinstance(row['text'], str) else row['text']
            if isinstance(row['image'], dict) and 'bytes' in row['image']:
                image_bytes = row['image']['bytes']
            else:
                image_bytes = row['image']
        
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            question = text_data.get('question', '')
            chosen_answer = text_data.get('chosen', '')
            
            if '<image>' not in question:
                question = f"<image>\n{question}"
            
            conversations = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": chosen_answer}
            ]
            
            data_list.append({
                'image': image,
                'conversations': conversations,
                'idx': idx,
                'metainfo': {
                    'origin_dataset': 'RLHF-V',
                    'origin_split': 'train',
                    'origin_idx': idx,
                    'image_id': f'rlhfv_{idx}',
                }
            })
            
        except Exception as e:
            continue

    return data_list