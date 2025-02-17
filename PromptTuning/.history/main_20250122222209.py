import copy

import numpy as np
import wandb
from peft import prepare_model_for_int8_training
# from utils.utils import prepare_model_for_int8_training
from transformers import Trainer, HfArgumentParser, set_seed, AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from distill_training_args import ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments
import warnings
import logging
import os
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from myutils import build_compute_metrics_fn, print_trainable_parameters, build_rouge_metrics_fn
#1AdhEVjrfvg-P6cpSBWJgQd0Dyp6K_c7N
from functools import partial
from src.model_distill import DistillModel
from transformers.modeling_utils import unwrap_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.dataset_distill import data_collator_with_padding, DistillDataset, C4PretrainDataset
from Mytrainer import MyTrainer as myTrainer, MyEarlyStop
from transformers.trainer_callback import EarlyStoppingCallback
from collections import defaultdict
from torch.utils.data import Subset
import evaluate
import json
from collections import OrderedDict
import torch
from transformers.generation.configuration_utils import GenerationConfig
# from src.BigModel import MyGPT2LMHeadModel


macro_f1_fn = partial(f1_score, average="macro")
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
rouge_eval_metric = evaluate.load("rouge")


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_API_KEY"] = 'KEY'
os.environ["WANDB_MODE"] = "offline"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if os.getenv('WANDB_PROJECT', None):
    os.environ["WANDB_PROJECT"] = "MetaICLExp"
def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setattr(data_args, "is_fid", model_args.is_fid)
    setattr(data_args, "demo_batch_count", model_args.demo_batch_count)
    setattr(model_args, "add_same_task_count", data_args.add_same_task_count)
    setattr(model_args, "t_demo_batch_count", data_args.t_demo_batch_count)
    setattr(model_args, "do_train", training_args.do_train)
    setattr(data_args, "is_pre_train", training_args.is_pre_train)
    setattr(data_args, "virtual_demo_len", model_args.virtual_demo_len)
    setattr(data_args, "is_t_demo", model_args.is_t_demo)
    if data_args.eval_catched_file is None:
        setattr(data_args, "eval_catched_file", data_args.catched_file)
    # data_args.is_debug = training_args.is_debug
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    os.environ["WANDB_PROJECT"] = training_args.wandb_project_name
    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    # logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.train_seed)

    special_tokens = []
    if "llama"  in model_args.s_model_name_or_path.lower():
        student_tokenizer = LlamaTokenizer.from_pretrained(model_args.s_model_name_or_path)
    else:
        student_tokenizer = AutoTokenizer.from_pretrained(
            model_args.s_model_name_or_path,
        )
    if "llama" in model_args.model_name_or_path.lower():
        teacher_tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            use_fast="gpt" in model_args.model_name_or_path,
            add_prefix="gpt" in model_args.model_name_or_path,
        )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.pad_token_id = teacher_tokenizer.eos_token_id

    # model_args.model_name_or_path,
    local_llm_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path
    )
    if hasattr(local_llm_config, "pad_token_id"):
        local_llm_config.pad_token_id = local_llm_config.eos_token_id


    local_llm = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=local_llm_config,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=not (training_args.is_train_llm or training_args.is_deep_speed)
            ),
            low_cpu_mem_usage=not training_args.is_deep_speed,
        )
    if model_args.t_state_dict_path is not None:
        t_state_dict = torch.load(model_args.t_state_dict_path, map_location='cpu')
        t_state_dict = {key.replace("local_llm_model.", "") if "local_llm_model" in key else key: value for key, value in
                        t_state_dict.items()}
        local_llm.load_state_dict(t_state_dict)

    ########### Generation relevant configuration for SNLI dataset ################
    if training_args.is_nli:
        generation_config = GenerationConfig(max_new_tokens=data_args.max_new_tokens)
        training_args.generation_config = generation_config

    task = data_args.task
    with open(os.path.join("config", task + ".json"), "r") as f:
        config = json.load(f)
    train_task_list = config['train']
    if data_args.is_unseen_domain and "unseen_domain_test" in config:
        task_list = config["unseen_domain_test"]
    else:
        task_list = config['test']
    if training_args.is_target_train:
        train_task_list = task_list
    if training_args.is_debug:
        task_list = ['glue-rte', 'glue-mrpc']
        train_task_list = task_list

    task_dict_is_clf = {}
    if training_args.is_nli is False:
        for here_task in task_list:
            with open("./config/tasks/{}.json".format(here_task),'r') as f1:
                a = json.load(f1)
                task_dict_is_clf[here_task] = a['task_type'] == 'classification'


    if training_args.is_train_llm is False and training_args.is_deep_speed is False:
        local_llm = prepare_model_for_int8_training(local_llm, use_gradient_checkpointing=not training_args.is_no_gradient_check)
    train_dataset = None
    task_idx_map = None
    if training_args.is_nli:
        here_dataset_class = SNIDataset
    else:
        here_dataset_class = DistillDataset

    if training_args.do_train:
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            if training_args.is_c4_pretrain:
                train_dataset = C4PretrainDataset(
                    data_args, tokenizer=teacher_tokenizer
                )
            else:
                train_dataset = here_dataset_class(data_args, split='train',
                                           seed=training_args.seed,
                                           K=data_args.k,
                                           teacher_tokenizer=teacher_tokenizer,
                                           student_tokenizer=student_tokenizer,
                                           virtual_demo_len=model_args.virtual_demo_len,
                                           is_target_train=training_args.is_target_train,
                                           is_debug=training_args.is_debug,
                                           mother_task_list=train_task_list)
        task_idx_map = getattr(train_dataset, "task_idx_map", None)
        if training_args.max_train_samples is not None:
            max_train_samples = training_args.max_train_samples
            train_dataset = Subset(train_dataset, indices=range(training_args.max_train_samples))
        else:
            max_train_samples = len(train_dataset)
    print(train_dataset.all_dataset[291040])

    if training_args.do_eval:
        with training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset_dict = OrderedDict()
            print("WE ARE PREPARING LOADING {}".format(task_list))

            for eval_task in task_list:
                # directly testing on test dataset to see upperbound
                eval_dataset = here_dataset_class(data_args, split='eval',
                                               seed=training_args.seed,
                                               K=data_args.k,
                                               teacher_tokenizer=teacher_tokenizer,
                                               student_tokenizer=student_tokenizer,
                                               virtual_demo_len=model_args.virtual_demo_len,
                                               is_debug=training_args.is_debug,
                                               task_list=[eval_task],
                                                mother_task_list=task_list,
                                               task_idx_map=task_idx_map if model_args.is_prompt_tuning else None,
                                                is_eval=True
                                               )
                if len(eval_dataset) == 0:
                    print("{} is 0".format(eval_task))
                    continue
                eval_dataset_dict[eval_task] = eval_dataset
            print("ATTENTION TASK LIST {}".format(eval_dataset_dict))



    num_tasks = train_dataset.num_tasks if train_dataset is not None else None

    model_fn = DistillModel
    model = model_fn(local_llm_model=local_llm,
                         model_args=model_args,
                         num_tasks=num_tasks,
                         is_train_llm=training_args.is_train_llm
                         )
    model.print_trainable_parameters()
    data_collator_fn = partial(data_collator_with_padding, t_tokenizer=teacher_tokenizer, is_fp16=training_args.fp16)

    if training_args.is_8bit_adam:
        import bitsandbytes as bnb
        from transformers.trainer_pt_utils import get_parameter_names
        import torch.nn as nn
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer_kwargs = {
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
        }
        optimizer_kwargs["lr"] = training_args.learning_rate
        adam_bnb_optim = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )

    trainer = myTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        data_collator=data_collator_fn,
        eval_dataset=eval_dataset_dict if training_args.do_eval else None,
        compute_metrics=build_compute_metrics_fn("validation") if training_args.is_nli is False else build_rouge_metrics_fn("validation", teacher_tokenizer, metric=rouge_eval_metric),
        optimizers=(adam_bnb_optim, None) if training_args.is_8bit_adam else (None, None),
        callbacks=[MyEarlyStop(
            early_stopping_patience=training_args.early_stop_threhold)] if training_args.do_train and training_args.load_best_model_at_end else None
    )
    trainer.set_task_dict_is_clf(task_dict_is_clf)
    if trainer.is_world_process_zero() and training_args.is_debug is False:
        wandb.init(settings=wandb.Settings(init_timeout=120))
        if training_args.is_deep_speed is False:
            wandb.config.update(vars(training_args))
        wandb.config.update(vars(data_args))
        wandb.config.update(vars(model_args))

    if training_args.do_train:
        train_result = trainer.train(training_args.checkpoint_dir)
        model.save_pretrained(training_args.output_dir)
        metrics = train_result.metrics
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        logger.info("**** Evaluation ****")
        if training_args.eval_seeds_list is None:
            eval_seeds_list = "13,21,42,87,100"
        else:
            eval_seeds_list = training_args.eval_seeds_list


        # task_list = config['glue']
        eval_result_log = defaultdict(list)
        all_avg_acc = []
        all_avg_f1 = []
        all_avg_final = []
        all_avg_rouge = []
        logger.info(f"WE are evaluating {task_list}")

        for eval_task in task_list:
            for step, seed in enumerate(eval_seeds_list.split(",")):
                with training_args.main_process_first(desc="Evaluation on {}".format(task)):
                    test_dataset = here_dataset_class(data_args,
                                                  split='eval',
                                                  seed=seed,
                                                  K=16,
                                                  teacher_tokenizer=teacher_tokenizer,
                                                  student_tokenizer=student_tokenizer,
                                                  virtual_demo_len=model_args.virtual_demo_len,
                                                  is_debug=training_args.is_debug,
                                                  is_target_train=False,
                                                  task_list=[eval_task],
                                                  mother_task_list=task_list,
                                                  task_idx_map=task_idx_map if model_args.is_prompt_tuning else None
                                                  )
                if len(test_dataset) == 0:
                    print(f"{eval_task} data is empty")
                    continue
                trainer.compute_metrics = build_compute_metrics_fn(task) if training_args.is_nli is False else build_rouge_metrics_fn(task, teacher_tokenizer, metric=rouge_eval_metric)
                if training_args.is_nli:
                    eval_result = trainer.evaluate(eval_dataset=test_dataset, **vars(generation_config))
                else:
                    eval_result = trainer.evaluate(eval_dataset=test_dataset)
                eval_result = {key+"_{}".format(eval_task): value for key, value in eval_result.items()}
                if trainer.is_world_process_zero():
                    for key, value in eval_result.items():
                        eval_result_log[key].append(value)
                    log_eval_result = {key: value for key, value in eval_result.items()}
                    wandb.log(log_eval_result, step=step)
                trainer.log_metrics("eval", eval_result)

        if trainer.is_world_process_zero():
            for key, value in eval_result_log.items():
                avg = np.mean(value)
                std = np.std(value)
                logger.info({f"AVG_{key}": avg, f"STD_{key}": std})
                wandb.log({f"AVG_{key}": avg, f"STD_{key}": std})
                if "acc" in key:
                    all_avg_acc.append(avg)
                elif "f1" in key:
                    all_avg_f1.append(avg)
                elif "rougeL" in key:
                    all_avg_rouge.append(avg)
                for task_name, is_classification in task_dict_is_clf.items():
                    if task_name in key.lower():
                        if is_classification:
                            if "f1" in key:
                                all_avg_final.append(avg)
                        else:
                            if "acc" in key:
                                all_avg_final.append(avg)
            final_avg_acc = np.mean(all_avg_acc)
            final_avg_f1 = np.mean(all_avg_f1)
            final_avg_final = np.mean(all_avg_final)
            final_avg_rouge = np.mean(all_avg_rouge)
            wandb.log({f"Final/AVG_acc":final_avg_acc, f"Final/AVG_f1":final_avg_f1, f"Final/Avg_Final":final_avg_final, 'Final/Avg_rougeL':final_avg_rouge})



                # trainer.save_metrics("eval", eval_result)

if __name__ == '__main__':
    main()
