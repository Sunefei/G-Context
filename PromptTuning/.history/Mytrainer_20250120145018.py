from transformers import Trainer
from typing import Optional
import os
from transformers.utils import logging, is_safetensors_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
import torch

if is_safetensors_available():
    import safetensors.torch
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model

logger = logging.get_logger(__name__)
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
# from transformers.trainer import PREFIX_CHECKPOINT_DIR, ShardedDDPOption, is_torch_tpu_available
import warnings
from transformers.trainer_pt_utils import reissue_pt_warnings
import random
import numpy as np
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
import wandb
from typing import Dict, Union, Any, List, Tuple
from transformers.trainer_callback import EarlyStoppingCallback
import torch.nn as nn
from transformers.trainer import nested_detach
from transformers.trainer_seq2seq import Seq2SeqTrainer
# from transformers.deepspeed import is_deepspeed_zero3_enabled



def metric_acc_agg(metrics, task_dict_is_clf):
    acc_list = []
    f1_list = []
    final_list = []
    rouge_list = []
    for key, value in metrics.items():
        if "acc" in key and 'FinalAcc' not in key:
            task_key = key.replace("eval_", "").replace("_acc", "")
            is_classification = task_dict_is_clf[task_key] if task_dict_is_clf is not None else False
            if is_classification is False:
                final_list.append(value)
            acc_list.append(value)
    for key, value in metrics.items():
        if "f1" in key and 'FinalF1' not in key:
            task_key = key.replace("eval_", "").replace("_f1", "")
            is_classification = task_dict_is_clf[task_key] if task_dict_is_clf is not None else False
            if is_classification is True:
                final_list.append(value)
            f1_list.append(value)
    for key, value in metrics.items():
        if "rougeL" in key:
            rouge_list.append(value)
    return np.mean(acc_list), np.mean(f1_list), np.mean(final_list), np.mean(rouge_list)


class MyEarlyStop(EarlyStoppingCallback):

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics.get('is_final', False) is False:
            return
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None and kwargs.get("is_final", False):
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return
        else:
            logger.info("EARLY STOPPING WORKING Value {}".format(metric_value))
            logger.info("EARLY STOPPING WORKING Keys {}".format([key for key, value in metrics.items() if 'f1' in key]))

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class MyTrainer(Trainer):
    def set_task_dict_is_clf(self, task_dict_is_clf):
        self.task_dict_is_clf = task_dict_is_clf
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint

        # Determine the new best metric / best model checkpoint

        super()._save_checkpoint(model, trial)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # agg metrics
            task_dict_is_clf = getattr(self, 'task_dict_is_clf', None)
            metric_values = metric_acc_agg(metrics, task_dict_is_clf)
            if self.is_world_process_zero():
                wandb.log({"Dev/" + key: value for key, value in metrics.items()}, commit=True)
                wandb.log({"Dev/FinalAcc": metric_values[0], "Dev/FinalF1": metric_values[1], "Dev/FinalFinal": metric_values[2], "Dev/FinalRougeL":metric_values[3]}, commit=True)
            metrics['eval_FinalF1'] = metric_values[1]
            metrics['eval_FinalAcc'] = metric_values[0]
            metrics['eval_FinalFinal'] = metric_values[2]
            metrics['eval_FinalRougeL'] = metric_values[3]
            metrics['is_final'] = True
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics=metrics)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(metrics[self.args.metric_for_best_model])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
