#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None, tune_clip=False):
        super(LlavaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower") and mm_vision_tower is not None:
            self.vision_tower = [CLIPVisionModel.from_pretrained(mm_vision_tower)]
            if tune_clip:
                self.vision_tower = self.vision_tower[0]

        if hasattr(config, "use_mm_proj"):
            mm_projector_type = getattr(config, 'mm_projector_type', 'linear')
            
            if mm_projector_type == 'mlp2x_gelu':
                self.mm_projector = nn.Sequential(
                    nn.Linear(config.mm_hidden_size, config.hidden_size),
                    nn.GELU(),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            else:
                self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                      pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower
    
        vision_tower_model = None
        vision_model_loaded = False
        local_clip_path = "path/to/clippatch"
        
        if os.path.exists(local_clip_path):
                config_file = os.path.join(local_clip_path, "config.json")
                vision_tower_model = CLIPVisionModel.from_pretrained(
                    local_clip_path,
                    local_files_only=True
                )
                vision_model_loaded = True

        
        if not vision_model_loaded and os.path.exists(str(vision_tower)):
                vision_tower_model = CLIPVisionModel.from_pretrained(
                    vision_tower,
                    local_files_only=True
                )
                vision_model_loaded = True
        
        if not vision_model_loaded:
            if hasattr(self, 'vision_tower') and self.vision_tower is not None:
                if isinstance(self.vision_tower, list):
                    vision_tower_model = self.vision_tower[0]
                else:
                    vision_tower_model = self.vision_tower
                vision_model_loaded = True
        
        if not vision_model_loaded:

            from transformers import CLIPVisionConfig

            vision_config = CLIPVisionConfig(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_channels=3,
                image_size=336, 
                patch_size=14,
                projection_dim=768,
                layer_norm_eps=1e-5,
            )
            
            vision_tower_model = CLIPVisionModel(vision_config)
            vision_model_loaded = True
 
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vision_tower_model = vision_tower_model.to(device=device, dtype=torch.float16)
        vision_tower_model.requires_grad_(False) 
        
        self.vision_tower = [vision_tower_model]
 
        vision_config = vision_tower_model.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
        
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        
        if not hasattr(self, 'mm_projector') or self.mm_projector is None:
            mm_projector_type = getattr(self.config, 'mm_projector_type', 'linear')
            
            if mm_projector_type == 'mlp2x_gelu':
                self.mm_projector = nn.Sequential(
                    nn.Linear(vision_config.hidden_size, self.config.hidden_size),
                    nn.GELU(),
                    nn.Linear(self.config.hidden_size, self.config.hidden_size)
                )
            else:
                self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
        else:
            if isinstance(self.mm_projector, nn.Sequential):
                first_layer = self.mm_projector[0]
                if first_layer.in_features != vision_config.hidden_size:
                    self.mm_projector = nn.Sequential(
                        nn.Linear(vision_config.hidden_size, self.config.hidden_size),
                        nn.GELU(),
                        nn.Linear(self.config.hidden_size, self.config.hidden_size)
                    )
            else:
                if self.mm_projector.in_features != vision_config.hidden_size:
                    mm_projector_type = getattr(self.config, 'mm_projector_type', 'linear')
                    if mm_projector_type == 'mlp2x_gelu':
                        self.mm_projector = nn.Sequential(
                            nn.Linear(vision_config.hidden_size, self.config.hidden_size),
                            nn.GELU(),
                            nn.Linear(self.config.hidden_size, self.config.hidden_size)
                        )
                    else:
                        self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
        
        if pretrain_mm_mlp_adapter is not None:
            try:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                
                if isinstance(self.mm_projector, nn.Sequential):
                    state_dict = {}
                    for key, value in mm_projector_weights.items():
                        if 'mm_projector.0' in key:
                            new_key = key.replace('model.mm_projector.0', '0')
                            state_dict[new_key] = value
                        elif 'mm_projector.2' in key:
                            new_key = key.replace('model.mm_projector.2', '2')
                            state_dict[new_key] = value
                    
                    self.mm_projector.load_state_dict(state_dict, strict=False)
                else:
                    # 单层Linear
                    self.mm_projector.load_state_dict(
                        {k.split('.')[-1]: v for k, v in mm_projector_weights.items()},
                        strict=False
                    )
            except Exception as e:
               import traceback
               traceback.print_exc()
        
        image_processor = CLIPImageProcessor(
            size={"shortest_edge": vision_config.image_size},
            crop_size={"height": vision_config.image_size, "width": vision_config.image_size},
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            image_mean=[0.48145466, 0.4578275, 0.40821073],
            image_std=[0.26862954, 0.26130258, 0.27577711],
            resample=3,  # PIL.Image.BICUBIC
        )
        
        return dict(
            image_processor=(image_processor, image_processor),
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            # Clamp input_ids before embedding to avoid CUDA out-of-bounds assertion.
            # IMAGE_TOKEN_INDEX = -200 (and any im_patch/im_start/im_end tokens that
            # exceed vocab_size) must not be passed directly to embed_tokens.
            # These positions will be overwritten by vision features below, so the
            # temporary embedding value for them is irrelevant.
            safe_input_ids = input_ids.clamp(min=0, max=self.embed_tokens.num_embeddings - 1)
            inputs_embeds = self.embed_tokens(safe_input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            if isinstance(vision_tower, (list, )):
                vision_tower = vision_tower[0]  # HACK: for FSDP
                with torch.no_grad():
                    if type(images) is list:
                        # variable length images
                        image_features = []
                        for image in images:
                            image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                            select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                            select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                            image_feature = select_hidden_state[:, 1:]
                            image_features.append(image_feature)
                    else:
                        image_forward_outs = vision_tower(images, output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                        image_features = select_hidden_state[:, 1:]
            else:
                # print(f'tune_clip', flush=True)
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]

            projector_dtype = next(self.mm_projector.parameters()).dtype
            
            if type(images) is list:
                image_features = [
                    self.mm_projector(image_feature.to(dtype=projector_dtype))[0] 
                    for image_feature in image_features
                ]
            else:
                image_features = image_features.to(dtype=projector_dtype)
                image_features = self.mm_projector(image_features)
            try:
                _n_patches = (vision_tower.config.image_size // vision_tower.config.patch_size) ** 2
            except Exception:
                _n_patches = 576  
            dummy_image_features = torch.zeros(
                _n_patches, 1024,
                device=inputs_embeds.device,
                dtype=projector_dtype
            )
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, mm_vision_tower=None, tune_clip=False):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config, mm_vision_tower=mm_vision_tower, tune_clip=tune_clip)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        
        if hasattr(self.model, 'vision_tower') and self.model.vision_tower is not None:
            if isinstance(self.model.vision_tower, list):
                vision_config = self.model.vision_tower[0].config
            else:
                vision_config = self.model.vision_tower.config
        
        vision_config.use_im_start_end = mm_use_im_start_end
   
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))
  
        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], 
                special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            
            vision_config.im_start_token, vision_config.im_end_token = \
                tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
            
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data
                
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
                
            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            if pretrain_mm_mlp_adapter:
                    mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                    embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                    assert num_new_tokens == 2
                    
                    if input_embeddings.shape == embed_tokens_weight.shape:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                    elif embed_tokens_weight.shape[0] == num_new_tokens:
                        input_embeddings[-num_new_tokens:] = embed_tokens_weight
                    else:
                        raise ValueError(
                            f"Unexpected embed_tokens_weight shape. "
                            f"Pretrained: {embed_tokens_weight.shape}. "
                            f"Current: {input_embeddings.shape}. "
                            f"Numer of new tokens: {num_new_tokens}."
                        )
        
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]



if os.environ.get('MUFFIN_SKIP_AUTO_REGISTER', '0') != '1':
    try:
        AutoConfig.register("llava", LlavaConfig, exist_ok=True)
    except ValueError:
        pass 
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)