from transformers import BartModel, RobertaModel, GPT2Model
import torch
from typing import Optional, Tuple, Union, List
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput, \
    BaseModelOutputWithPoolingAndCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel
import torch
from typing import Optional, Tuple, Union



def add_init_prompt_weight(model_config, model, virtual_demo_len, model_args):
    prompt_init_embedding = nn.Embedding(virtual_demo_len, model_config.hidden_size, dtype=model.dtype)
    if model_args.virtual_demo_init == "random":
        if hasattr(model_config, "initializer_range"):
            init_range = model_config.initializer_range
        else:
            try:
                init_range = model_config.init_std
            except:
                init_range = 0.1
        prompt_init_embedding.weight.data.normal_(mean=0.0, std=init_range)
    elif model_args.virtual_demo_init == 'vocab':
        rand_id = torch.randint(100, model_config.vocab_size, (virtual_demo_len,)).long()
        a = model.get_input_embeddings()
        print("***ATTENTION Original EMBEDDING SHAPE {}".format(a))
        rand_id = rand_id.unsqueeze(0)
        rand_emb = model.get_input_embeddings()(rand_id).squeeze()
        print("***ATTENTION EMBEDDING SHAPE {}".format((rand_emb.shape)))
        prompt_init_embedding = prompt_init_embedding.from_pretrained(rand_emb, freeze=False)
    init_prompt_weight = prompt_init_embedding.weight
    print("****ATTENTION WEIGHT {}".format(init_prompt_weight.dtype))
    # print(init_prompt_weight.dtype)
    return init_prompt_weight


class MyGPT2(GPT2Model):

    def make_seperate_attention_mask(self, attention_mask, demo_labels_token_ids, virtual_demo_len,
                                     label_virtual_demo_len):
        # batch_size, 1, 1, seq_len => batch_size, num_heads, tgt_seq_len, seq_len
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = attention_mask.to(dtype=self.dtype)
        demo_labels_token_ids = demo_labels_token_ids.to(dtype=self.dtype)
        attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)
        # mask out the label_virtual_demo_len
        # batch_size, seq_len
        demo_labels_token_ids = torch.cat((demo_labels_token_ids,
                                           demo_labels_token_ids.new_zeros(demo_labels_token_ids.shape[0],
                                                                           virtual_demo_len - label_virtual_demo_len),
                                           demo_labels_token_ids.new_ones(demo_labels_token_ids.shape[0],
                                                                          label_virtual_demo_len)), dim=1)
        demo_input_token_ids = 1 - demo_labels_token_ids
        # mask out the last 2nd index
        # attention_mask[:, :, -label_virtual_demo_len:, :] *= demo_labels_token_ids[:, None, None, :]
        attention_mask[:, :, -label_virtual_demo_len:, :] *= demo_labels_token_ids[:, None, None, :]
        if self.config.is_expand_demo_input_mask:
            attention_mask[:, :, -virtual_demo_len:-label_virtual_demo_len, :] *= demo_input_token_ids[:, None, None, :]
        # mask out the last index
        # attention_mask[:, :, :, -label_virtual_demo_len:] *= demo_labels_token_ids[:, None, :, None]

        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        return attention_mask

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            demo_token_type_ids=None,
            **kwargs
    ):
        isz = input_ids.shape[0] // self.config.demo_batch_count

        if hasattr(self, "init_prompt_weight"):
            distill_embeds = self.init_prompt_weight.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
            inputs_embeds = torch.cat([self.get_input_embeddings()(input_ids), distill_embeds], dim=1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(distill_embeds.shape[0], distill_embeds.shape[1])], dim=-1)
        else:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if self.config.expand_s_attention_mask:
            here_attention_mask = self.make_seperate_attention_mask(attention_mask,
                                                                    demo_token_type_ids,
                                                                    virtual_demo_len=self.config.virtual_demo_len,
                                                                    label_virtual_demo_len=self.config.label_virtual_demo_len
                                                                    )
        else:
            here_attention_mask = attention_mask
        last_hidden_states = super().forward(
            None,
            past_key_values,
            here_attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            output_hidden_states=True).last_hidden_state

        new_hidden_states = last_hidden_states.reshape(isz, -1, last_hidden_states.shape[-1])
        new_attention_mask = attention_mask.reshape(isz, -1)
        if hasattr(self, "init_prompt_weight"):
            distill_embeds_shape = self.init_prompt_weight.shape
            new_hidden_states = new_hidden_states[:, -distill_embeds_shape[0]:, :]
            new_attention_mask = new_attention_mask[:, -distill_embeds_shape[0]:]
        return new_hidden_states, new_attention_mask
