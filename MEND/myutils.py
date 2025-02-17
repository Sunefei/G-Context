import logging

from transformers.trainer_utils import EvalPrediction
from typing import Callable, Dict, Optional, List, Tuple
import numpy as np
from functools import partial
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import json
import os
import string
import re as regex
import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk



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
    # def postprocess_text(preds, labels):
    #     preds = [pred.strip() for pred in preds]
    #     labels = [label.strip() for label in labels]
    #     return preds, labels
    
    # copy from ceil used to calculate EM
    def normalize_answer(s):
        def remove_articles(text):
            return regex.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def exact_match_score(prediction, ground_truth):
        label = normalize_answer(ground_truth)
        predict = normalize_answer(prediction)
        return label == predict


    def max_em(prediction, ground_truths):
        return max([exact_match_score(prediction, gt) for gt in ground_truths])


    def compute_metrics_fn(eval_preds):
        inputs = eval_preds.inputs # None
        labels = eval_preds.label_ids
        preds = eval_preds.predictions

        # 确保长度一样
        assert(len(labels) == len(preds))
        
        # 标签为索引，直接从数据集里load
        label_path = "/data/syf/gsy/MEND/catched_data/label3.torch"
        label_data = load_from_disk(label_path)
        
        # 过滤标签数据
        label = []
        pred = []
        for i in range(labels.shape[0]):  # 遍历批量中的每一条数据
            index = int(labels[i][0])
            label.append(label_data[index]['output'])
            filtered_pred = preds[i][(preds[i] > 0)] # 保留非负的 token ID
            pred.append(torch.tensor(filtered_pred))
        #转化为tensor
        pred = pad_sequence(pred, batch_first=True, padding_value=tokenizer.pad_token_id)

        # 解码成文本
        decoded_preds = tokenizer.batch_decode(pred, skip_special_tokens=True)
        
        # 计算 Exact Match
        eval_results = []
        for pred, gold_answers in zip(decoded_preds, label):
            eval_results.append(max_em(pred, gold_answers))
        
        result1 = {"em": sum(eval_results) / len(decoded_preds)}
        print("Exact match1:", result1)
        logging.info("**** Exact Match: {} ****".format(task_name))
        
        # result2 = 0
        # for i in range(len(label[0])):
        #     for j in range(len(label[i])):
        #         result2 += metric.compute(predictions=decoded_preds[i], 
        #                                   references=label[i][j],
        #                                   ignore_case=True,
        #                                   ignore_punctuation=True,
        #                                   regexes_to_ignore = [
        #                                         r"\b(a|an|the)\b",  # 忽略冠词（a, an, the）
        #                                         r"[^\w\s]",         # 忽略所有标点符号
        #                                         r"\s+",             # 忽略多余的空格（包括换行、制表符）
        #                                     ])
        # result2 = result2 / len(label[0])
        # print("Exact match2:", result2)
        
        return result1
        
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
