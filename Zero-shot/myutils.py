import logging

from transformers.trainer_utils import EvalPrediction
from typing import Callable, Dict, Optional, List, Tuple
import numpy as np
from functools import partial
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
import os
def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        logits = p.predictions
        preds = np.argmax(logits, axis=1).reshape(-1)
        try:
            label_ids = p.label_ids.reshape(-1)
        except:
            print(p.label_ids)
            exit(-1)
        cm = confusion_matrix(y_true=label_ids, y_pred=preds)
        logging.info("**** EVAL CONFUSION MATRIX: {} ****".format(task_name))
        print(cm)
        return {"f1": f1_score(y_true=label_ids, y_pred=preds, average='macro'),
                "acc": accuracy_score(y_true=label_ids, y_pred=preds)}

    return compute_metrics_fn

def build_rouge_metrics_fn(task_name, tokenizer, metric):
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def compute_metrics_fn(eval_preds):
        inputs = eval_preds.inputs
        labels = eval_preds.label_ids
        preds = eval_preds.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        if preds.shape[1] > inputs.shape[1]:
            preds = preds[:, inputs.shape[1]:]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        logging.info("**** EVAL CONFUSION MATRIX: {} ****".format(task_name))
        print(result)
        return result
    return compute_metrics_fn

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, is_split_all=False):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    for dataset in datasets:
        data_path = os.path.join("data", dataset,
                                 "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))
        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
    return data


def my_load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False,
              is_split_all=False,
              is_seed_all=False
                 ):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    if is_split_all:
        split_list = ['train','dev', 'test']
    else:
        split_list = [split]
    if is_seed_all:
        seed_list = [100, 13, 21, 42, 87]
    else:
        seed_list = [seed]
    for sd in seed_list:
        for sp in split_list:
            for dataset in datasets:
                data_path = os.path.join("data", dataset,
                                         "{}_{}_{}_{}.jsonl".format(dataset, k, sd, sp))
                with open(data_path, "r") as f:
                    for line in f:
                        dp = json.loads(line)
                        if is_null:
                            dp["input"] = "N/A"
                        dp['split'] = sp
                        dp['seed'] = sd
                        data.append(dp)
    return data
