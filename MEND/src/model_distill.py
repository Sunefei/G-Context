import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast, \
    SequenceClassifierOutput, CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
# from peft import prepare_model_for_int8_training
# from transformers import BitsAndBytesConfig
from transformers import LogitsProcessorList, TopPLogitsWarper, PreTrainedModel
from transformers import AutoTokenizer
import os
# from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers import AutoConfig
from .SmallModel import MyGPT2, add_init_prompt_weight
from collections import OrderedDict




class MyPooler(nn.Module):
    def __init__(self, s_hidden_size, t_hidden_size):
        super().__init__()
        self.dense = nn.Linear(s_hidden_size, t_hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        pooled_output = self.dense(flat_hidden_states)
        pooled_output = self.activation(pooled_output)
        nested_output = pooled_output.reshape(*hidden_states.shape[:-1], -1)
        return nested_output


class DistillModel(PreTrainedModel):
    def __init__(self, local_llm_model,
                 model_args,
                 config=None,
                 num_tasks=None,
                 is_train_llm=False,
                 ):
        super().__init__(local_llm_model.config)
        # add
        self.local_llm_model = local_llm_model
        # self.local_llm_model = local_llm_model.to("cuda:6")
        self.config = self.local_llm_model.config
        self.is_prompt_tuning = model_args.is_prompt_tuning
        self.virtual_demo_len = model_args.virtual_demo_len
        self.d_beta = model_args.d_beta
        self.d_eta = model_args.d_eta
        self.processors = LogitsProcessorList()
        self.processors.append(TopPLogitsWarper(0.95))
        self.d_tau = model_args.d_tau
        self.is_contrast = model_args.is_contrast
        self.is_t_demo = model_args.is_t_demo
        self.model_args = model_args
        self.is_train_llm = is_train_llm
        self.sm_pool_name = model_args.sm_pool_name
        self.is_kd = model_args.is_kd
        self.is_only_kd = model_args.is_only_kd
        self.loss_lambda = model_args.loss_lambda
        self.is_add_model_encode_demo = model_args.is_add_model_encode_demo
        self.is_fid = model_args.is_fid
        self.is_shared_prompt = model_args.is_shared_prompt
        # FID relevant parameters
        self.demo_batch_count = model_args.demo_batch_count
        # inner distill
        self.is_inner_distill = model_args.is_inner_distill
        self.ts_steps = model_args.ts_steps
        self.add_same_task_count = model_args.add_same_task_count
        self.t_demo_batch_count = model_args.t_demo_batch_count
        self.kl_topK = model_args.kl_topK
        self.do_train = model_args.do_train
        # fake lm_head to fool generate check
        # self.lm_head = self.local_llm_model.lm_head


        if is_train_llm is False:
            if model_args.is_prompt_tuning:
                self.num_tasks = num_tasks
                self.hypernet = self._init_prompt(d_len=self.virtual_demo_len, num_tasks=self.num_tasks)
            else:
                config = AutoConfig.from_pretrained(model_args.s_model_name_or_path)
                s_tokenizer = AutoTokenizer.from_pretrained(model_args.s_model_name_or_path)
                config.mask_token_id = s_tokenizer.mask_token_id
                config.demo_batch_count = self.demo_batch_count
                config.virtual_demo_len = self.virtual_demo_len
                config.label_virtual_demo_len = self.model_args.label_virtual_demo_len
                config.expand_s_attention_mask = self.model_args.expand_s_attention_mask
                config.is_expand_demo_input_mask = self.model_args.is_expand_demo_input_mask
                if 'bart' in model_args.s_model_name_or_path.lower():
                    self.hypernet = MyFiDBart.from_pretrained(model_args.s_model_name_or_path,
                                                              config=config)
                    self.is_s_roberta = False
                elif 'roberta' in model_args.s_model_name_or_path.lower():
                    config.is_s_reuse_cache = model_args.is_s_reuse_cache
                    config.is_ed = model_args.is_ed
                    if self.model_args.is_fid:
                        config.is_decoder = True
                        if model_args.is_s_reuse_cache is False and model_args.is_ed is False:
                            config.add_cross_attention = True
                    self.hypernet = MyFiDRoBERTa.from_pretrained(model_args.s_model_name_or_path, config=config,
                                                                 add_pooling_layer=False)
                    self.is_s_roberta = True
                elif "gpt2" in model_args.s_model_name_or_path.lower():
                    self.hypernet = MyGPT2.from_pretrained(model_args.s_model_name_or_path,
                                                           config=config,
                                                           # load_in_8bit=not self.do_train
                                                           )
                elif "t5" in model_args.s_model_name_or_path.lower():
                    # TODO: Implementation of t5 model here.
                    raise NotImplementedError
                else:
                    self.hypernet = AutoModel.from_pretrained(model_args.s_model_name_or_path,
                                                           config=config,
                                                           # load_in_8bit=not self.do_train
                                                           )
                self.hypernet.adap_pooler = MyPooler(self.hypernet.config.hidden_size,
                                                     self.local_llm_model.config.hidden_size) \
                    if self.hypernet.config.hidden_size != self.local_llm_model.config.hidden_size else None
                if self.model_args.is_init_prompt_weight:
                    self.hypernet.init_prompt_weight = add_init_prompt_weight(self.hypernet.config, model=self.hypernet,
                                                                              virtual_demo_len=self.virtual_demo_len,
                                                                              model_args=model_args)

                if self.model_args.s_state_dict_path is not None:
                    s_state_dict = torch.load(self.model_args.s_state_dict_path, map_location='cpu')
                    s_state_dict = {key.replace("hypernet.", "") if "hypernet" in key else key: value for key, value in
                            s_state_dict.items()}
                    self.hypernet.load_state_dict(s_state_dict)

    def _init_prompt(self, d_len, num_tasks):
        prompt_length = d_len * num_tasks
        hypernet = nn.Embedding(d_len * num_tasks, self.local_llm_model.config.hidden_size)
        if self.model_args.virtual_demo_init == "random":
            hypernet.weight.data.normal_(mean=0.0, std=self.local_llm_model.config.initializer_range)
        elif self.model_args.virtual_demo_init == 'vocab':
            rand_id = torch.randint(100, self.config.vocab_size, (prompt_length,)).long()
            rand_emb = self.embed_encode(rand_id)
            hypernet = hypernet.from_pretrained(rand_emb, freeze=False)
        # reshape the weight for hypernet
        weight = hypernet.weight.data
        weight = weight.reshape(num_tasks, -1)
        hypernet = nn.Embedding(*weight.shape).from_pretrained(weight, freeze=False)
        return hypernet

    def get_task_prompt_embed(self, task_index):
        if self.is_shared_prompt:
            task_index = torch.zeros_like(task_index)
        prompt_embed = self.hypernet(task_index)
        prompt_embed = prompt_embed.reshape(task_index.shape[0], self.virtual_demo_len, -1)
        return prompt_embed

    def query_logits_distill(self,
                                   s_logits,
                                   t_logits,
                                   s_input_attention_mask,
                                   t_input_attention_mask,
                                   ):
        loss_fn = nn.KLDivLoss(reduction="batchmean")
        # extract the s_logits
        flat_s_logits = s_logits.reshape(-1, s_logits.shape[-1]).contiguous()
        flat_t_logits = t_logits.reshape(-1, t_logits.shape[-1]).contiguous()
        flat_t_attention_mask = t_input_attention_mask.reshape(-1, 1).contiguous().bool()
        flat_s_attention_mask = s_input_attention_mask.reshape(-1, 1).contiguous().bool()
        select_s_logits = torch.masked_select(flat_s_logits, flat_s_attention_mask).reshape(-1, flat_s_logits.shape[-1])
        select_t_logits = torch.masked_select(flat_t_logits, flat_t_attention_mask).reshape(-1, flat_t_logits.shape[-1])

        kl_loss = loss_fn(
            F.log_softmax(select_s_logits/self.model_args.temperature, dim=1),
            F.softmax(select_t_logits/self.model_args.temperature, dim=1)
        )
        return kl_loss


    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: torch.LongTensor = None,
            teacher_input_ids: torch.LongTensor = None,
            teacher_attention_mask: Optional[torch.Tensor] = None,
            teacher_token_type_ids=None,
            demo_ids=None,
            demo_attention_mask=None,
            num_class=-1,
            label_candidates: Optional[torch.LongTensor] = None,
            label_candidates_attention_mask: Optional[torch.LongTensor] = None,
            label_candidates_token_type_ids: Optional[torch.LongTensor] = None,
            add_input_ids=None,
            add_attention_mask=None,
            add_token_type_ids=None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            clf_label=None,
            task_idx=None,
            teacher_demo_token_type_ids=None,
            is_consecutive=None,
            real_num_class_mask=None,
            demo_token_type_ids=None,
            past_key_values=None,
            s_demo_count=None,
            **kwargs,
    ):
        is_generation = False
        if clf_label is not None:
            batch_size = clf_label.shape[0]
        else:
            batch_size = input_ids.shape[0]
        if (
                self.training is False and self.is_t_demo) or self.is_train_llm or self.is_kd or self.model_args.is_meta_gradient_match or self.model_args.is_query_state_match or self.model_args.is_query_kl_loss:
            if self.is_train_llm:
                t_llm_output = self.local_llm_model(input_ids=teacher_input_ids,
                                                    attention_mask=teacher_attention_mask,
                                                    use_cache=True,
                                                    output_hidden_states=True,
                                                    output_attentions=output_attentions
                                                    )

            else:
                # in-context learning inference or teacher
                # no need for gradient
                with torch.no_grad():
                    t_llm_output = self.local_llm_model(input_ids=teacher_input_ids,
                                                        attention_mask=teacher_attention_mask,
                                                        use_cache=True,
                                                        output_hidden_states=True,
                                                        output_attentions=output_attentions
                                                        )
                    
            t_logits = t_llm_output.logits
            attentions = t_llm_output.attentions


        if ((
                    self.training is False and self.is_t_demo) or self.is_train_llm) is False or self.is_kd or self.model_args.is_meta_gradient_match or self.is_inner_distill or self.model_args.is_query_state_match or self.model_args.is_query_kl_loss:
            # ATTENTION: get the student logits.
            distill_attention_mask = None
            if self.is_prompt_tuning:
                distill_embeds = self.get_task_prompt_embed(task_idx)
            else:
                if self.model_args.is_encoder_debug:
                    distill_embeds = self.local_llm_model.get_input_embeddings()(demo_ids)
                    distill_attention_mask = distill_attention_mask
                else:
                    distill_embeds, distill_attention_mask = self.hypernet(input_ids=demo_ids,
                                                                       attention_mask=demo_attention_mask,
                                                                       sm_pool_name=self.sm_pool_name,
                                                                       demo_token_type_ids=demo_token_type_ids)

                    if self.hypernet.adap_pooler is not None:
                        distill_embeds = self.hypernet.adap_pooler(distill_embeds)

                # expand for evaluation
                if self.training is False and is_generation is False:
                    assert num_class > 0
                    distill_embeds = distill_embeds.unsqueeze(1).repeat(1, num_class, 1, 1).reshape(
                        -1, *distill_embeds.shape[1:]
                    )
                    if distill_attention_mask is not None:
                        distill_attention_mask = distill_attention_mask.repeat(1, num_class, 1).reshape(
                            -1, *distill_attention_mask.shape[1:]
                        )

            if distill_attention_mask is None:
                distill_attention_mask = attention_mask.new_ones(distill_embeds.shape[0], distill_embeds.shape[1])

            # TODO: Prepare the 2D attention mask to highlight the special embeddings for outputs
            # pretraining loss
            if self.model_args.is_seq2seq_pretrain and self.training:
                input_ids = teacher_input_ids
                attention_mask = teacher_attention_mask
                token_type_ids = teacher_attention_mask
                distill_embeds = distill_embeds.unsqueeze(1).repeat(1, self.model_args.t_demo_batch_count,
                                                                    1, 1).reshape(-1, *distill_embeds.shape[1:])
                distill_attention_mask = distill_attention_mask.unsqueeze(1).repeat(1,
                                                                                    self.model_args.t_demo_batch_count,
                                                                                    1).reshape(-1,
                                                                                               distill_attention_mask.shape[
                                                                                                   -1])

            input_ids_embeds = self.local_llm_model.get_input_embeddings()(input_ids)
            # concatenate embeds
            if self.model_args.is_no_s_embeds:
                input_embeds_s = input_ids_embeds
                attention_mask_s = attention_mask
            else:
                input_embeds_s = torch.cat((distill_embeds, input_ids_embeds), dim=1)
                attention_mask_s = torch.cat([distill_attention_mask, attention_mask], dim=-1)
            # share the past_key_values for other class labels

            s_llm_output = self.local_llm_model(
                inputs_embeds=input_embeds_s,
                attention_mask=attention_mask_s,
                use_cache=True,
                output_hidden_states=True,
                output_attentions=output_attentions
            )
            # downstream tasks' objective functions
            s_logits = s_llm_output.logits
            past_key_values = s_llm_output.past_key_values
            attentions = s_llm_output.attentions
            # if is_generation:
            #     return CausalLMOutputWithCrossAttentions(loss = loss,past_key_values=past_key_values, logits=s_logits)
            # remove the distilled sentence hidden states
            if self.model_args.is_no_s_embeds is False:
                s_logits = s_logits[:, distill_embeds.shape[1]:]

        if self.training:
            if self.is_train_llm:
                # MetaICL
                train_logits = t_logits
                train_gen_labels = torch.clone(teacher_input_ids)
                train_token_type_ids = teacher_token_type_ids
                train_attention_mask = teacher_attention_mask
            else:
                train_logits = s_logits
                train_gen_labels = torch.clone(input_ids)
                train_token_type_ids = token_type_ids
                train_attention_mask = attention_mask

        # Flatten the tokens
        loss_logits = None
        loss = None
        if self.training and self.model_args.is_only_auxiliary_loss is False:
            labels_mask = train_token_type_ids[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            # Shift so that tokens < n predict n
            shift_logits = train_logits[..., :-1, :].contiguous()
            shift_labels = train_gen_labels[..., 1:].contiguous()
            shift_train_attention_mask = train_attention_mask[:, 1:].contiguous()
            if is_consecutive is not None:
                # only subsect of a batch requires to calculate teacher forcing generation loss.
                is_consecutive = is_consecutive.bool().squeeze()
                shift_logits = shift_logits[is_consecutive]
                shift_labels = shift_labels[is_consecutive]
                labels_mask = labels_mask[is_consecutive]
            shift_labels = torch.where((labels_mask * shift_train_attention_mask) == 1,
                                       shift_labels,
                                       torch.ones_like(shift_labels) * -100)
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss_logits = None
        elif self.training is False:
            if self.is_t_demo:
                shift_logits = t_logits[:, :-1].contiguous()
                shift_labels = torch.where(teacher_token_type_ids == 1, teacher_input_ids,
                                           torch.ones_like(teacher_input_ids) * -100)[:, 1:].contiguous()
                here_token_type_ids = teacher_token_type_ids[:, 1:].contiguous()
            else:
                shift_logits = s_logits[:, :-1].contiguous()
                shift_labels = torch.where(token_type_ids == 1, input_ids,
                                           torch.ones_like(input_ids) * -100)[:, 1:].contiguous()
                here_token_type_ids = token_type_ids[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1)
            )
            loss = loss.view(batch_size * num_class, -1)
            # agg loss for classification
            # shape: batch_size, num_class for classification
            loss_logits = -1 * (torch.sum(loss, dim=1) / torch.sum(here_token_type_ids, dim=1)).reshape(batch_size,
                                                                                                        num_class)
            # mask out the loss_logits
            if real_num_class_mask is not None:
                loss_logits = loss_logits.masked_fill((1-real_num_class_mask).bool(), -10000)
            loss = torch.mean(loss)

        # Auxilary loss
        if self.training:
            # knowledge distillation between teacher and student
            # batch_size, seq_length, vocab_size
            if self.model_args.is_query_kl_loss:
                s_input_token_type_ids = attention_mask
                t_input_token_type_ids = (1 - teacher_demo_token_type_ids) * teacher_attention_mask
                aux_loss = self.query_logits_distill(
                    s_logits=s_logits,
                    t_logits=t_logits,
                    s_input_attention_mask=s_input_token_type_ids,
                    t_input_attention_mask=t_input_token_type_ids
                )
            else:
                aux_loss = 0
            if aux_loss is not None and type(aux_loss) is not int and torch.isnan(aux_loss):
                raise FloatingPointError("aux_loss is nan")
            if self.model_args.is_only_auxiliary_loss:
                loss = aux_loss
            else:
                loss += self.loss_lambda * aux_loss

        if loss is not None and torch.isnan(loss):
            raise FloatingPointError("loss is nan")

        return SequenceClassifierOutput(
            loss=loss,
            logits=loss_logits,
            attentions=attentions
        )

    @torch.no_grad()
    def save_pretrained(self, save_directory, safe_serialization=False, **kwargs):
        if "state_dict" in kwargs.keys():
            kwargs.pop("state_dict")
        # we only save the distillation model
        if self.is_train_llm:
            state_dict = self.local_llm_model.state_dict()
            state_dict = OrderedDict({"local_llm_model." + key: value for key, value in state_dict.items()})
            self.local_llm_model.save_pretrained(save_directory, safe_serialization, state_dict=state_dict, **kwargs)
        else:
            state_dict = self.hypernet.state_dict()
            state_dict = OrderedDict({"hypernet." + key: value for key, value in state_dict.items()})
            if isinstance(self.hypernet, PreTrainedModel):
                self.hypernet.save_pretrained(
                    save_directory, safe_serialization, state_dict=state_dict, **kwargs
                )
            else:
                torch.save(state_dict, save_directory + "/pytorch_model.bin")

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        return {
            **kwargs,
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

