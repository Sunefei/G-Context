from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments, Seq2SeqTrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    s_model_name_or_path: str = field(
        default="gpt2",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    virtual_demo_len: int = field(
        default=5,
    )
    d_beta: float = field(
        default=0.5
    )
    d_eta: float = field(
        default=0.1,
    )
    d_tau: float = field(
        default=0.05,
    )
    is_contrast: bool = field(
        default=False
    )
    is_kd: bool = field(
        default=False
    )
    is_t_demo: bool = field(
        default=False
    )
    is_prompt_tuning: bool = field(
        default=False
    )
    is_shared_prompt: bool = field(
        default=False
    )
    virtual_demo_init: str = field(
        default="random"
    )
    sm_pool_name: str=field(
        default=None,
    )
    loss_lambda: float = field(
        default=1.
    )
    is_only_kd: bool = field(
        default=False
    )
    is_add_model_encode_demo: bool = field(
        default=False
    )
    demo_batch_count: int = field(
        default=1
    )
    is_fid: bool = field(
        default=False
    )
    is_s_reuse_cache: bool = field(
        default=False,
        metadata={"help": "Only used for encoder demonstration distillation model. "
                          "It will use the past_key_values from demonstrations when set to True, "
                          "otherwise will use encoder_hidden_states"}
    )

    ts_steps: int = field(
        default=1
    )

    is_inner_distill: bool = field(
        default=False
    )

    lr_inner_distill: float = field(
        default=1e-3
    )

    s_state_dict_path: str = field(
        default=None
    )

    t_state_dict_path: str = field(
        default=None
    )

    kl_topK: int = field(
        default=-1
    )
    is_ed: bool = field(
        default=False
    )
    is_encoder_debug: bool = field(
        default=False
    )
    is_seq2seq_pretrain: bool = field(
        default=False
    )
    is_only_auxiliary_loss: bool = field(
        default=False
    )
    is_meta_gradient_match: bool = field(
        default=False
    )
    is_query_state_match: bool = field(
        default=False
    )
    is_init_prompt_weight: bool = field(
        default=False
    )
    is_no_s_embeds: bool = field(
        default=False
    )
    last_match_layers_hidden:int = field(
        default=-100
    )
    label_virtual_demo_len: int = field(
        default=1,
    )
    expand_s_attention_mask: bool = field(
        default=False
    )
    is_expand_demo_input_mask: bool = field(
        default=False
    )
    is_query_kl_loss: bool = field(
        default=False
    )
    temperature: float = field(
        default=1
    )
    is_t5_hyper_encoder: bool = field(
        default=False
    )
    vis_attenion_size: int = field(
        default=5
    )

@dataclass
class DynamicDataTrainingArguments:
    """
    Arguments for dynamic training.
    """
    demo_count: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )
    t_max_length: int = field(
        default=1024
    )
    s_max_length: int = field(
        default=512
    )
    max_length_per_example: int = field(
        default=256
    )
    dataset_name: str = field(
        default=None
    )
    catched_file: str = field(
        default=None
    )
    eval_catched_file: str = field(
        default=None
    )
    overwrite_cache: bool = field(
        default=False,
    )
    task: str = field(
        default="class_to_class",
    )

    k: int = field(
        default=16384,
    )

    test_k: int = field(
        default=16
    )

    method: str = field(
        default="direct",
    )
    is_not_hf: bool = field(
        default=False
    )
    is_unseen_domain: bool = field(
        default=False
    )

    similar_fn: str = field(
        default=None
    )

    is_100_shot:bool = field(
        default=False
    )
    add_same_task_count: int = field(
        default=0
    )
    t_demo_count: int = field(
        default=4
    )
    t_demo_batch_count: int = field(
        default=0
    )

    student_input_percent:float = field(
        default=0.8
    )

    shuffle_input_qm_prob: float = field(
        default=0
    )
    is_demo_no_label: bool = field(
        default=False
    )
    is_query_prefix_random_label: bool = field(
        default=False
    )
    split_query_pretrain_pro: float = field(
        default=0
    )
    is_demo_itself: bool = field(
        default=False,
        metadata={"help":"debug usage"}
    )
    is_demo_only_label: bool = field(
        default=False,
        metadata={"help": "debug usage"}
    )
    use_new_options_map: bool = field(
        default=False
    )
    is_include_test_example: bool = field(
        default=False
    )
    max_new_tokens: int= field(
        default=128
    )
    max_target_length: int = field(
        default=128
    )
    is_nli_2pos: bool = field(
        default=False
    )
    nli_pos_example_length: int = field(
        default=256
    )
    is_ins_metaicl: bool = field(
        default=False
    )
    is_only_prepare_data: bool = field(
        default=False
    )
    is_t5_seq2seq: bool = field(
        default=False
    )
    is_eval_search: bool = field(
        default=False
    )
    is_encoder_input: bool = field(
        default=False
    )
    is_wrong_label: bool = field(
        default=False
    )
    is_random_label: bool = field(
        default=False
    )
    special_tag: str = field(
        default=None
    )


@dataclass
class DynamicTrainingArguments(Seq2SeqTrainingArguments):
    # Turn off train/test
    do_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )

    is_debug: bool = field(
        default=False
    )

    train_seed: int = field(
        default=1
    )

    is_target_train: bool = field(
        default=False
    )

    max_train_samples: int = field(
        default=None
    )

    eval_seeds_list: str = field(
        default=None
    )

    wandb_project_name: str = field(
        default="MetaICLExp"
    )

    is_train_llm: bool = field(
        default=False
    )

    is_deep_speed: bool = field(
        default=False
    )

    is_8bit_adam: bool = field(
        default=False
    )

    is_pre_train: bool = field(
        default=False
    )

    is_c4_pretrain: bool = field(
        default=False
    )

    is_nli: bool = field(
        default=False
    )
    early_stop_threhold: int = field(
        default=5
    )
    is_no_gradient_check: bool = field(
        default=False
    )
    checkpoint_dir:str=field(
        default=None
    )