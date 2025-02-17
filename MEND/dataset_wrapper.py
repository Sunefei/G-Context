import os
import logging
from datasets import load_from_disk
from datasets import Dataset as hu_Dataset
from datasets import concatenate_datasets
from datasets import load_dataset

def process_sst5_dataset(sst5_path, is_train, is_target_train):
    all_dataset = load_from_disk(sst5_path)
    tmp = []
    label =  ["very negative", "negative", "neutral", "positive", "very positive"]
    if is_train and not is_target_train:
        train_dataset = all_dataset['train']
        for data in train_dataset:
            tmp.append(create_metaicl_format(data,
                                            task = 'sst5', 
                                            input = data['text'],
                                            output = data['label_text'],
                                            label_options= label,
                                            split = 'meta-train'))

    else:
        for split in ['train', 'validation', 'test']:
            dataset = all_dataset[split]
            if split == 'validation':
                    split = 'dev'
            for data in dataset:
                tmp.append(create_metaicl_format(data,
                                            task = 'sst5', 
                                            input = data['text'],
                                            output = data['label_text'],
                                            label_options= label,
                                            split = split))

    return hu_Dataset.from_dict({
        "task": [meta["task"] for meta in tmp],
        "input": [meta["input"] for meta in tmp],
        "output": [meta["output"] for meta in tmp],
        "options": [meta["options"] for meta in tmp],
        "seed": [meta["seed"] for meta in tmp],
        "split": [meta["split"] for meta in tmp]
    })

def process_swag_dataset(swag_path, is_train, is_target_train):
    all_dataset = load_from_disk(swag_path)
    tmp = []
    if is_train and not is_target_train:
        train_dataset = all_dataset['train']
        for data in train_dataset:
            label = data['label'] 
            tmp.append(create_metaicl_format(data,
                                            task = 'swag', 
                                            input = data["ctx"],
                                            output = data["endings"][int(label) if label and label.isdigit() and int(label) < len(data["endings"]) else ""],
                                            label_options= data["endings"],
                                            split = 'meta-train'))

    else:
        for split in ['validation', 'train', 'validation']:
            dataset = all_dataset[split]
            if split == 'validation':
                if len(tmp) == 0:
                    split = 'test'
                else:
                    split = 'dev'
            for data in dataset:
                label = data['label']
                tmp.append(create_metaicl_format(data,
                                                task = 'swag', 
                                                input = data["ctx"],
                                                output = data["endings"][int(label) if label and label.isdigit() and int(label) < len(data["endings"]) else ""],
                                                label_options= data["endings"],
                                                split = split))

    return hu_Dataset.from_dict({
        "task": [meta["task"] for meta in tmp],
        "input": [meta["input"] for meta in tmp],
        "output": [meta["output"] for meta in tmp],
        "options": [meta["options"] for meta in tmp],
        "seed": [meta["seed"] for meta in tmp],
        "split": [meta["split"] for meta in tmp]
    })


def process_mrpc_dataset(is_train, is_target_train):
    all_dataset = load_dataset("glue", 'mrpc')
    if is_train and not is_target_train:
        dataset = all_dataset['train']
        return process_mrpc(dataset,'meta-train')
    else:
        train = process_mrpc(all_dataset['train'], 'train')
        dev = process_mrpc(all_dataset['validation'], 'dev')
        test = process_mrpc(all_dataset['validation'], 'test')
        return concatenate_datasets([train, dev, test])
        
        
def process_mnli_dataset(is_train, is_target_train):
    all_dataset = load_dataset("glue", 'mnli')
    if is_train and not is_target_train:
        dataset = all_dataset['train']
        return process_mnli(dataset,'meta-train')
    else:
        train = process_mnli(all_dataset['train'], 'train')
        concate = concatenate_datasets([all_dataset['validation_matched'], all_dataset['validation_mismatched']])
        dev = process_mnli(concate, 'dev')
        test = process_mnli(concate, 'test')
        return concatenate_datasets([train, dev, test])

def create_metaicl_format(data, task, input, output, label_options, split):
    return {
        "task": task, 
        "input": input,  
        "output": output,  
        "options": label_options,  # all possible choice
        "seed": '100',  
        "split": split 
    }
    
def process_mrpc(dataset, split_name):
    all_labels = ["No", "Yes"]
    dataset = dataset.map(lambda x: {**x, 'output': all_labels[x['label']]})
    dataset = dataset.map(lambda x: {**x, 'input': 
        "{sentence1} Can we say \"{sentence2}\"? ".format(
            sentence1=x["sentence1"],
            sentence2=x["sentence2"])
    })
    
    return hu_Dataset.from_dict({
        'task': ['mrpc'] * len(dataset),
        'input': dataset['input'],
        'output': dataset['output'],
        'options': [all_labels] * len(dataset),
        'seed': ['100'] * len(dataset),
        'split': [split_name] * len(dataset)
    })

def process_mnli(dataset, split_name):
    all_labels = ["Yes", "Maybe", "No"]
    dataset = dataset.map(lambda x: {**x, 'output': all_labels[x['label']]})
    dataset = dataset.map(lambda x: {**x, 'input': 
        "{premise} Can we say \"{hypothesis}\"? ".format(
            hypothesis=x["hypothesis"],
            premise=x["premise"])
    })
    
    return hu_Dataset.from_dict({
        'task': ['mnli'] * len(dataset),
        'input': dataset['input'],
        'output': dataset['output'],
        'options': [all_labels] * len(dataset),
        'seed': ['100'] * len(dataset),
        'split': [split_name] * len(dataset)
    })


