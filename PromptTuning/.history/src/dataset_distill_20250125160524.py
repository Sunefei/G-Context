import copy
import math

import datasets
import tqdm
from torch.utils.data import Dataset
import numpy as np
from itertools import chain
import os
import json
from datasets import Dataset as hu_Dataset
from datasets import load_from_disk
from datasets import DatasetDict
import pandas as pd
from transformers import AutoTokenizer
from functools import partial
import logging
from collections import defaultdict
from datasets import load_dataset
import torch
from time import time

NLIOPTIONSMAP = [" yes", " no", " maybe"]

def flatten(inputs):
    return list(chain.from_iterable(inputs))

class DistillDataset(Dataset):
    def __init__(self,
                 data_args,
                 split,
                 seed,
                 K,
                 virtual_demo_len=10,
                 is_debug=False,
                 teacher_tokenizer=None,
                 student_tokenizer=None,
                 is_target_train=False,
                 task_list=None,
                 is_eval=False,
                 task_idx_map=None,
                 is_pre_train=False,
                 mother_task_list=None
                 ):
        super().__init__()
        # only handle on preselected task
        # if is_target_train or split != 'train':
        assert mother_task_list is not None
        self.is_eval = is_eval
        self.task_list = task_list
        self.is_debug = is_debug
        self.data_args = data_args
        self.max_length_per_example = data_args.max_length_per_example
        self.is_target_train = is_target_train
        self.method = data_args.method
        self.demo_count = data_args.demo_count
        self.s_max_length = data_args.s_max_length
        self.t_max_length = data_args.t_max_length
        self.teacher_tokenizer = teacher_tokenizer
        self.t_name = teacher_tokenizer.name_or_path.split("/")[-1]
        self.student_tokenizer = student_tokenizer
        self.s_name = student_tokenizer.name_or_path.split("/")[-1]
        self.task_dict = defaultdict(list)
        self.seed_dict = defaultdict(list)
        self.test_task_dict = defaultdict(list)
        # self.output_template = " Output is: "
        self.is_train = split == "train"
        self.is_s_no_bos = True
        self.s_sep_token_id = self.student_tokenizer.sep_token_id
        self.s_mask_token_id = self.student_tokenizer.mask_token_id
        self.s_bos_token_id = self.student_tokenizer.bos_token_id
        # for llama "\n"
        self.t_add_newlines = "llama" in self.t_name
        # ATTENTION: This only works for llama relevant models.
        self.t_sep_token_id = 13
        self.t_bos_token_id = self.teacher_tokenizer.pad_token_id  if "t5" in self.t_name else self.teacher_tokenizer.bos_token_id
        self.virtual_demo_len = virtual_demo_len
        self.is_not_hf = self.data_args.is_not_hf
        self.seed = seed
        self.eval_K = K
        self.similar_fn = data_args.similar_fn
        self.is_fid = data_args.is_fid
        self.demo_batch_count = data_args.demo_batch_count
        self.add_same_task_count = data_args.add_same_task_count
        self.t_demo_count = data_args.t_demo_count
        self.t_demo_batch_count = data_args.t_demo_batch_count
        self.is_pre_train = is_pre_train
        self.is_demo_no_label = data_args.is_demo_no_label
        self.is_query_prefix_random_label = data_args.is_query_prefix_random_label
        self.new_options_token_map = teacher_tokenizer(NLIOPTIONSMAP, add_special_tokens=False)['input_ids']
        self.is_test = False

        assert self.demo_count <= self.eval_K
        if self.is_train and self.is_target_train is False:
            catched_file = data_args.catched_file
        else:
            catched_file = data_args.eval_catched_file
        logging.info("*** ATTENTION We are LOADING {} *****".format(catched_file))
        if os.path.exists(catched_file) is False or data_args.overwrite_cache:
            logging.info("*** ATTENTION We are Creating {} *****".format(catched_file))
            if self.data_args.is_ins_metaicl:
                all_dataset = load_dataset("bigheiniuJ/InstructEvalMetaICLAll")
            else:
                all_dataset = load_dataset("bigheiniuJ/EvalMetaICLAll")
            if self.is_train and self.is_target_train is False:
                self.all_dataset = all_dataset['meta_train']
            elif data_args.is_100_shot:
                self.all_dataset = all_dataset['meta_eval_100shot']
            else:
                self.all_dataset = all_dataset['meta_eval']
            if self.is_debug:
                # self.all_dataset = self.all_dataset.filter(lambda x: x['task'] == 'glue-sst2' or x['task'] == 'glue-mrpc' or x['task'] == 'glue-rte')
                self.all_dataset = self.all_dataset.filter(
                    lambda x: x['task'] == 'glue-mrpc' or x['task'] == 'glue-rte')
                # self.all_dataset = self.all_dataset.filter(lambda x: x['task'] == 'spider')
            # filter the dataset
            task_set = set(mother_task_list)
            self.all_dataset = self.all_dataset.filter(lambda x: x['task'] in mother_task_list)
            logging.info("THERE ARE {} INSTANCES, {}".format(len(self.all_dataset), self.is_train))
            t_tokenize_fn = partial(self.tokenize_input, tokenizer=self.teacher_tokenizer, prefix=self.t_name)
            s_tokenize_fn = partial(self.tokenize_input, tokenizer=self.student_tokenizer, prefix=self.s_name)
            self.all_dataset = self.all_dataset.map(t_tokenize_fn, batched=True, desc="Teacher Tokenization")
            self.all_dataset = self.all_dataset.map(s_tokenize_fn, batched=True, desc="Student Tokenization")
            self.all_dataset = self.all_dataset.map(partial(self.prerocess_tokenized_input, prefix=self.t_name),
                                                    desc="Teacher Preprocess")
            self.all_dataset = self.all_dataset.map(partial(self.prerocess_tokenized_input, prefix=self.s_name),
                                                    desc="Student Preprocess")
            self.all_dataset = self.all_dataset.map(self.get_class_labels, desc="GET Class Labels")
            self.all_dataset.save_to_disk(catched_file)
        else:
            self.all_dataset = load_from_disk(catched_file)

        # speed up the filter operation.
        if os.path.exists(catched_file + "/ori_task_index_map.torch") is False or data_args.overwrite_cache:
            self.original_task_dict = defaultdict(list)
            for index, i in enumerate(self.all_dataset['task']):
                self.original_task_dict[i].append(index)
            torch.save(self.original_task_dict, catched_file + "/ori_task_index_map.torch")
        else:
            self.original_task_dict = torch.load(catched_file + "/ori_task_index_map.torch")

        if os.path.exists(catched_file + "/ori_seed_index_map.torch") is False or data_args.overwrite_cache:
            self.ori_seed_dict = defaultdict(list)
            for index, i in enumerate(self.all_dataset['seed']):
                self.ori_seed_dict[i].append(index)
            torch.save(self.ori_seed_dict, catched_file + "/ori_seed_index_map.torch")
        else:
            self.ori_seed_dict = torch.load(catched_file + "/ori_seed_index_map.torch")
        if data_args.is_only_prepare_data:
            exit(0)

        # load task is_classification
        self.task_dict_is_clf = {}
        for task_i in self.original_task_dict.keys():
            config_file = "config/tasks/{}.json".format(task_i)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = int(config["task_type"] == "classification")
            self.task_dict_is_clf[task_i] = is_classification

        # only work in interested task and seed
        # this is only designed to improve efficiency
        if self.task_list is not None:
            task_select_index = flatten([self.original_task_dict[i] for i in self.task_list])
            # select based on seed
            seed_select_index = self.ori_seed_dict[str(seed)]
            task_select_index = list(set(task_select_index).intersection(set(seed_select_index)))
            self.all_dataset = self.all_dataset.select(task_select_index)

        if self.is_train and self.is_target_train is False:
            self.target_dataset = self.all_dataset
        else:
            # choose the evaluation dataset
            if is_eval:
                # validation for hyper-parameter choose or early stop
                self.target_dataset = self.all_dataset.filter(lambda x: x['split'] == 'dev')
            else:
                # test dataset for prediction
                self.target_dataset = self.all_dataset.filter(lambda x: x['split'] == 'test')
                self.is_test = True
            # choose the demonstration dataset
            self.all_dataset = self.all_dataset.filter(lambda x: x['split'] == 'train')
            if self.is_target_train:
                self.target_dataset = self.all_dataset

        # get the latest update
        for index, i in enumerate(self.all_dataset['task']):
            self.task_dict[i].append(index)

        for index, i in enumerate(self.target_dataset['task']):
            self.test_task_dict[i].append(index)

        for index, i in enumerate(self.all_dataset['seed']):
            # for evaluation, the demonstration is selected based on seed and task
            self.seed_dict[i].append(index)

        # make sure train and test they got the same order.
        if task_idx_map is None:
            task_set = set(self.target_dataset['task'])
            self.num_tasks = len(task_set)
            self.task_idx_map = {i: index for index, i in enumerate(sorted(task_set))}
        else:
            self.task_idx_map = task_idx_map
            self.num_tasks = len(task_idx_map)

        # task based similarity KNN creation
        task_relevant_dict = None
        # only works for KNN model
        self.task_relevant_dict = task_relevant_dict

        if self.is_pre_train:
            self.glue_instruct = json.load(open("config/debug_glue_ins.json", 'r'))
            self.glue_instruct = {key: self.teacher_tokenizer(key)['input_ids'] for key, value in
                                  self.glue_instruct.items()}

    def assign_task_index(self, batch):
        batch['task_index'] = list(range(len(batch)))
        return batch

    def prerocess_tokenized_input(self, dp, prefix):
        input_tokens = dp[prefix + 'input_tokens']
        output_tokens = dp[prefix + 'output_tokens']
        if "task" in dp:
            if (dp["task"].startswith("inst:piqa") or dp["task"].startswith("inst:yahoo_answers_topics")) and \
                    len(input_tokens) + len(output_tokens) + 2 > self.max_length_per_example:
                input_tokens = input_tokens[:self.max_length_per_example // 2]
                output_tokens = output_tokens[:self.max_length_per_example // 2 - 2]

            elif len(input_tokens) >= self.max_length_per_example - 2 - len(output_tokens):
                if dp["task"].startswith("inst:") and len(input_tokens) < len(output_tokens):
                    output_tokens = output_tokens[:self.max_length_per_example - 2 - len(input_tokens)]
                else:
                    input_tokens = input_tokens[:self.max_length_per_example - 2 - len(output_tokens)]

        assert len(input_tokens) + len(output_tokens) + 2 <= self.max_length_per_example, \
            (dp.get("task", None), len(input_tokens), len(output_tokens), self.max_length_per_example)

        if self.method == "direct":
            dp[prefix + 'input_tokens'] = input_tokens
            dp[prefix + 'output_tokens'] = output_tokens
        elif self.method == "channel":
            dp[prefix + 'input_tokens'] = output_tokens
            dp[prefix + 'output_tokens'] = input_tokens

        return dp

    def tokenize_input(self, batch, tokenizer, prefix):
        input_tokens = tokenizer([i for i in batch['input']],
                                 add_special_tokens=False, truncation=True)['input_ids']
        # adding whitespace for output and options gpt2 based language model
        options = [i if len(i) > 0 else ["NONE"] for i in batch['options']]
        option_length = [len(i) for i in options]
        cum_option_length = np.cumsum([0] + option_length)
        flat_options = list(chain.from_iterable(options))
        if "llama" not in tokenizer.name_or_path.lower():
            batch_output = [" " + i for i in batch['output']]
            flat_options = [" " + i for i in flat_options]
            second_input_tokens = \
            tokenizer([" " + i for i in batch['input']], add_special_tokens=False, truncation=True)['input_ids']
        else:
            batch_output = batch['output']
            second_input_tokens = input_tokens
        encode_options = tokenizer(text=flat_options, add_special_tokens=False, truncation=True)
        encoded_output = tokenizer(text=batch_output, add_special_tokens=False, truncation=True)
        # nested_options
        option_tokens = [encode_options['input_ids'][cum_option_length[index]:cum_option_length[index + 1]]
                         for index in range(len(cum_option_length) - 1)]
        output_tokens = encoded_output['input_ids']
        tokenized_output = {"input_tokens": input_tokens,
                            "second_input_tokens": second_input_tokens,
                            "output_tokens": output_tokens,
                            "option_tokens": option_tokens}
        for key, value in tokenized_output.items():
            batch[prefix + key] = value
        return batch

    def get_class_labels(self, example):
        try:
            clf_label = example['options'].index(example['output'].strip())
        except:
            clf_label = -100
        example['clf_label'] = clf_label
        return example

    def make_one_demo(self, input, output, add_newlines):
        if add_newlines:
            input_ids = input + [self.t_sep_token_id] + output + [self.t_sep_token_id] * 2
            token_type_ids = [0] * (len(input) + 1) + [1] * (len(output)) + [0] * 2
        else:
            # for GPT2
            if self.is_demo_no_label:
                input_ids = input
                token_type_ids = [0] * len(input_ids)
            elif self.data_args.is_demo_only_label:
                input_ids = output
                token_type_ids = [1] * len(output)
            else:
                input_ids = input + output
                token_type_ids = [0] * len(input) + [1] * len(output)

        assert len(input_ids) == len(
            token_type_ids), f"ATTENTION ERROR ABOUT DIFFERENT LENGTH {len(input_ids)} vs {len(token_type_ids)}"
        return input_ids, token_type_ids

    def demo_concate(self, random_demo_idxes, prefix, max_length, demo_tokens, add_newlines):
        # specially designed for local llm demonstration distillation model (roberta-base)
        output_demo_tokens = copy.deepcopy(demo_tokens)
        output_demo_token_type_ids = [0] * len(demo_tokens)
        max_valid_length = max_length - self.max_length_per_example - 1
        demo_count = 0
        here_dataset = self.all_dataset if self.data_args.is_demo_itself is False else self.target_dataset
        for i in random_demo_idxes:
            if self.data_args.is_wrong_label:
                random_label = set(np.random.randint(0, len(here_dataset[i][prefix + 'option_tokens']), 2).tolist())
                current_label = here_dataset[i][prefix + 'option_tokens'].index(here_dataset[i][prefix + 'output_tokens'])
                here_label = random_label.difference(set(current_label))
                here_output = here_dataset[i][prefix+"option_tokens"][here_label]
            elif self.data_args.is_random_label:
                here_label = np.random.randint(0, len(here_dataset[i][prefix + 'option_tokens']), 1).tolist()[0]
                here_output = here_dataset[i][prefix + "option_tokens"][here_label]
            else:
                here_output = here_dataset[i][prefix + 'output_tokens']

            # print("input_tokens: ", here_dataset[i][prefix + 'input_tokens'])
            # print("output_tokens: ", here_output)
            buffer, buffer_demo_token_type_ids = self.make_one_demo(here_dataset[i][prefix + 'input_tokens'],
                                                                   here_output,
                                                                    add_newlines=add_newlines)
            # print("buffer: ", buffer)
            # print("token_type_ids: ", buffer_demo_token_type_ids)

            if len(buffer) + len(output_demo_tokens) < max_valid_length:
                output_demo_tokens += buffer
                output_demo_token_type_ids += buffer_demo_token_type_ids
            elif len(output_demo_tokens) + len(here_dataset[i][prefix + 'output_tokens']) < max_valid_length:
                rest_demo_ids, rest_demo_token_type_ids = \
                    self.make_one_demo(here_dataset[i][prefix + 'input_tokens'] \
                                           [:max_valid_length - len(output_demo_tokens) - len(
                        here_dataset[i][prefix + 'output_tokens']) - 1],
                                       here_dataset[i][prefix + 'output_tokens'],
                                       add_newlines=add_newlines)
                output_demo_tokens += rest_demo_ids
                output_demo_token_type_ids += rest_demo_token_type_ids

                break
            else:
                break
            demo_count += 1
        assert len(output_demo_tokens) <= max_valid_length, (len(output_demo_tokens), max_valid_length)
        assert len(output_demo_tokens) == len(output_demo_token_type_ids)
        return output_demo_tokens, demo_count, output_demo_token_type_ids

    def __len__(self):
        return len(self.target_dataset)

    def basic_getitem(self, item, demo_count=-100):
        dp = self.target_dataset[item]
        print("demo_count: ", demo_count)
        print("item index: ", item)
        print(dp)
        task = dp['task']
        task_idx = self.task_idx_map.get(task, 0)
        all_task_demo_idxes = self.task_dict[task]
        s_demo_inputs_tokens = []
        s_demo_inputs_tokens_list = []
        s_demo_token_type_ids_list = []
        t_demo_inputs_tokens_list = []
        s_demo_count_list = []
        teacher_input_tokens = []
        if demo_count > 0:
            for i in range(self.demo_batch_count):
                if self.task_relevant_dict is None:
                    permutation_idxes = np.random.permutation(len(all_task_demo_idxes))[:demo_count + 1].tolist()
                    random_demo_idxes = [all_task_demo_idxes[i] for i in permutation_idxes]
                    if item in random_demo_idxes:
                        random_demo_idxes.remove(item)
                    else:
                        random_demo_idxes = random_demo_idxes[:demo_count]
                else:
                    try:
                        random_demo_idxes = self.task_relevant_dict[task][item][:demo_count]
                    except:
                        print(f"### ERROR {task}, Index {item}, Index Keys {self.task_dict[task]}")
                        exit(-1)
                if self.data_args.is_demo_itself:
                    random_demo_idxes = [item] * len(random_demo_idxes)

                t_demo_inputs_tokens_here, t_demo_count, t_demo_token_type_ids = self.demo_concate(
                    random_demo_idxes, self.t_name, self.t_max_length, [], add_newlines=self.t_add_newlines
                )
                # print("t_demo_inputs_tokens_here: ", t_demo_inputs_tokens_here)
                # print("t_demo_token_type_ids: ", t_demo_token_type_ids)
                t_demo_inputs_tokens_list.append(t_demo_inputs_tokens_here)
                
                s_demo_inputs_tokens_here, s_demo_count, s_demo_token_type_ids = self.demo_concate(
                    random_demo_idxes, self.s_name, self.s_max_length, s_demo_inputs_tokens, add_newlines=False)
                s_demo_inputs_tokens_list.append(s_demo_inputs_tokens_here)
                s_demo_token_type_ids_list.append(s_demo_token_type_ids)
                s_demo_count_list.append(s_demo_count)
        else:
            t_demo_inputs_tokens_here = []
        # ATTENTION: only choose one for teacher
        teacher_input_tokens += t_demo_inputs_tokens_here

        teacher_demo_token_type_ids = [1] * len(teacher_input_tokens)
        teacher_input_tokens += dp[self.t_name + 'input_tokens']
        if self.data_args.is_include_test_example:
            s_demo_inputs_tokens_list = [i+dp[self.s_name+"input_tokens"] for i in s_demo_inputs_tokens_list]
            s_demo_token_type_ids_list = [i+[1]*len(dp[self.s_name+"input_tokens"]) for i in s_demo_token_type_ids_list]

        output_tokens = dp[self.t_name + 'output_tokens']
        output_token_type_ids = [1] * len(output_tokens)
        if len(teacher_input_tokens) + len(output_tokens) > self.t_max_length:
            # truncate from the left.
            teacher_input_tokens = teacher_input_tokens[
                                   len(teacher_input_tokens) + len(output_tokens) - self.t_max_length:]
            # add the seperated tokens for llama
            if self.t_add_newlines:
                # TODO: what if the student is also llama model???
                teacher_input_tokens += [self.t_sep_token_id]

        # demo_ids is designed for demonstration distillation model.
        demo_ids = s_demo_inputs_tokens_list
        demo_attention_mask = [[1] * len(i) for i in demo_ids]
        if self.is_train:
            input_ids = dp[self.t_name + 'input_tokens'] + output_tokens
            token_type_ids = [0] * len(dp[self.t_name + 'input_tokens']) + output_token_type_ids
            teacher_input_ids = teacher_input_tokens + output_tokens
            teacher_token_type_ids = [0] * len(teacher_input_tokens) + output_token_type_ids
            teacher_demo_token_type_ids += [0] * (len(teacher_token_type_ids) - len(teacher_demo_token_type_ids))
        else:
            # at inference time we did not append label with the input.
            input_ids = dp[self.t_name + 'input_tokens']
            token_type_ids = [0] * len(dp[self.t_name + 'input_tokens'])
            teacher_input_ids = teacher_input_tokens
            if self.is_query_prefix_random_label:
                # ATTENTION: add a random label to see ICL following ability
                random_label = np.random.randint(0, len(dp['options']), 1).item()
                teacher_input_ids = dp[self.t_name + "option_tokens"][random_label] + teacher_input_ids
            teacher_token_type_ids = [0] * len(teacher_input_tokens)
            teacher_demo_token_type_ids = None
        teacher_attention_mask = [1] * len(teacher_input_ids)
        attention_mask = [1] * len(input_ids)

        outpout = {"input_ids": input_ids,
                   "attention_mask": attention_mask,
                   "token_type_ids": token_type_ids,
                   "teacher_input_ids": teacher_input_ids,
                   "teacher_attention_mask": teacher_attention_mask,
                   "teacher_token_type_ids": teacher_token_type_ids,
                   "demo_ids": demo_ids,
                   "demo_attention_mask": demo_attention_mask,
                   "demo_token_type_ids": s_demo_token_type_ids_list,
                   "task_idx": [task_idx],
                   "teacher_demo_token_type_ids": teacher_demo_token_type_ids,
                   "s_demo_count": s_demo_count_list
                   }

        if self.is_train or self.is_target_train:
            return outpout
        else:
            t_len = len(teacher_input_ids)
            clf_label = dp['clf_label']
            label_candidates = dp[self.t_name + 'option_tokens']
            label_candidates = [i[:(self.t_max_length - len(i) - t_len)] for i in label_candidates]
            if self.data_args.use_new_options_map:
                if len(label_candidates) == 2:
                    label_candidates = self.new_options_token_map[:2]
                else:
                    # no, yes, maybe
                    label_candidates = [self.new_options_token_map[1], self.new_options_token_map[0],
                                        self.new_options_token_map[2]]
            label_candidates_attention_mask = [[1] * len(i) for i in label_candidates]
            label_candidates_token_type_ids = [[1] * len(i) for i in label_candidates]
            label_output = {"label_candidates": label_candidates,
                            "label_candidates_attention_mask": label_candidates_attention_mask,
                            "label_candidates_token_type_ids": label_candidates_token_type_ids,
                            "clf_label": [clf_label]
                            }
            output = {**outpout, **label_output}
            return output
    def basic_t5_getitem(self, item, demo_count=-100):
        dp = self.target_dataset[item]
        task = dp['task']
        task_idx = self.task_idx_map.get(task, 0)
        all_task_demo_idxes = self.task_dict[task]
        s_demo_inputs_tokens = []
        s_demo_inputs_tokens_list = []
        s_demo_token_type_ids_list = []
        t_demo_inputs_tokens_list = []
        s_demo_count_list = []
        teacher_input_tokens = []
        if demo_count > 0:
            for i in range(self.demo_batch_count):
                if self.task_relevant_dict is None:
                    permutation_idxes = np.random.permutation(len(all_task_demo_idxes))[:demo_count + 1].tolist()
                    random_demo_idxes = [all_task_demo_idxes[i] for i in permutation_idxes]
                    if item in random_demo_idxes:
                        random_demo_idxes.remove(item)
                    else:
                        random_demo_idxes = random_demo_idxes[:demo_count]
                else:
                    try:
                        random_demo_idxes = self.task_relevant_dict[task][item][:demo_count]
                    except:
                        print(f"### ERROR {task}, Index {item}, Index Keys {self.task_dict[task]}")
                        exit(-1)
                if self.data_args.is_demo_itself:
                    random_demo_idxes = [item] * len(random_demo_idxes)

                t_demo_inputs_tokens_here, t_demo_count, t_demo_token_type_ids = self.demo_concate(
                    random_demo_idxes, self.t_name, self.t_max_length, [], add_newlines=self.t_add_newlines
                )
                t_demo_inputs_tokens_list.append(t_demo_inputs_tokens_here)
                s_demo_inputs_tokens_here, s_demo_count, s_demo_token_type_ids = self.demo_concate(
                    random_demo_idxes, self.s_name, self.s_max_length, s_demo_inputs_tokens, add_newlines=False)
                s_demo_inputs_tokens_list.append(s_demo_inputs_tokens_here)
                s_demo_token_type_ids_list.append(s_demo_token_type_ids)
                s_demo_count_list.append(s_demo_count)
        else:
            t_demo_inputs_tokens_here = []
        teacher_input_tokens += t_demo_inputs_tokens_here
        if self.data_args.is_encoder_input:
            teacher_input_tokens += dp[self.t_name + 'input_tokens']
        if self.is_train:
            if self.data_args.is_encoder_input:
                decoder_input_ids = [self.t_bos_token_id] + dp[self.t_name+"output_tokens"]
                decoder_token_type_ids = [1] * len(decoder_input_ids)
            else:
                #
                decoder_input_ids = [self.t_bos_token_id] + dp[self.t_name + 'input_tokens'] + dp[self.t_name+"output_tokens"]
                decoder_token_type_ids = [0] * (len(dp[self.t_name + 'input_tokens']) + 1) + [1] * len(
                    dp[self.t_name + "output_tokens"])

        else:
            if self.data_args.is_encoder_input:
                decoder_input_ids = [self.t_bos_token_id]
                decoder_token_type_ids = [0] * len(decoder_input_ids)
            else:
                #
                decoder_input_ids = [self.t_bos_token_id] + dp[self.t_name + 'input_tokens']
                decoder_token_type_ids = [0] * len(decoder_input_ids)

        decoder_attention_mask = [1] * len(decoder_input_ids)
        # ATTENTION: we did not add the test input into the encoder
        # teacher_input_tokens += dp[self.t_name + 'input_tokens']

        if len(teacher_input_tokens) > self.t_max_length:
            teacher_input_tokens = teacher_input_tokens[:self.t_max_length]

        # demo_ids is designed for demonstration distillation model.
        demo_ids = s_demo_inputs_tokens_list
        demo_attention_mask = [[1] * len(i) for i in demo_ids]
        input_ids = dp[self.t_name + 'input_tokens']
        teacher_input_ids = teacher_input_tokens
        teacher_attention_mask = [1] * len(teacher_input_ids)
        attention_mask = [1] * len(input_ids)
        # input_ids is no usage here.
        outpout = {"input_ids": input_ids,
                   "attention_mask": attention_mask,
                   "teacher_input_ids": teacher_input_ids,
                   "teacher_attention_mask": teacher_attention_mask,
                   "demo_ids": demo_ids,
                   "demo_attention_mask": demo_attention_mask,
                   "demo_token_type_ids": s_demo_token_type_ids_list,
                   "task_idx": [task_idx],
                   "s_demo_count": s_demo_count_list,
                   "decoder_input_ids": decoder_input_ids,
                   "decoder_attention_mask": decoder_attention_mask,
                   "decoder_token_type_ids": decoder_token_type_ids
                   }

        if self.is_train or self.is_target_train:
            return outpout
        else:
            clf_label = dp['clf_label']
            label_candidates = dp[self.t_name + 'option_tokens']
            label_candidates = [i for i in label_candidates]
            label_candidates_attention_mask = [[1] * len(i) for i in label_candidates]
            label_candidates_token_type_ids = [[1] * len(i) for i in label_candidates]
            label_output = {"label_candidates": label_candidates,
                            "label_candidates_attention_mask": label_candidates_attention_mask,
                            "label_candidates_token_type_ids": label_candidates_token_type_ids,
                            "clf_label": [clf_label]
                            }
            output = {**outpout, **label_output}
            return output

    def get_multi_teacher(self, item):
        # TODO: Multi demo token type ids.
        dp = self.target_dataset[item]
        task = dp['task']
        task_idx = self.task_idx_map.get(task, 0)
        all_task_demo_idxes = self.task_dict[task]
        if self.is_s_no_bos:
            s_demo_inputs_tokens = []
        else:
            if self.is_fid is False:
                s_demo_inputs_tokens = [self.s_bos_token_id] + [self.s_mask_token_id] * self.virtual_demo_len
            else:
                s_demo_inputs_tokens = [self.s_bos_token_id]
        s_demo_inputs_tokens_list = []
        t_demo_inputs_tokens_list = []
        random_demo_idxes_list = []
        for i in range(self.demo_batch_count):
            if self.task_relevant_dict is None:
                permutation_idxes = np.random.permutation(len(all_task_demo_idxes))[:self.demo_count + 1].tolist()
                random_demo_idxes = [all_task_demo_idxes[i] for i in permutation_idxes]
                if item in random_demo_idxes:
                    random_demo_idxes.remove(item)
                else:
                    random_demo_idxes = random_demo_idxes[:self.demo_count]
            else:
                try:
                    random_demo_idxes = self.task_relevant_dict[task][item][:self.demo_count]
                except:
                    print(f"### ERROR {task}, Index {item}, Index Keys {self.task_dict[task]}")
                    exit(-1)
            random_demo_idxes_list.extend(random_demo_idxes)
            s_demo_inputs_tokens_here, s_demo_count, s_demo_token_type_ids = self.demo_concate(
                random_demo_idxes, self.s_name, self.s_max_length, s_demo_inputs_tokens, add_newlines=False)
            s_demo_inputs_tokens_list.append(s_demo_inputs_tokens_here)
        demo_ids = s_demo_inputs_tokens_list
        demo_attention_mask = [[1] * len(i) for i in demo_ids]
        output_tokens = dp[self.t_name + 'output_tokens']
        output_token_type_ids = [1] * len(output_tokens)
        input_ids = dp[self.t_name + 'input_tokens'] + output_tokens
        token_type_ids = [0] * len(dp[self.t_name + 'input_tokens']) + output_token_type_ids
        attention_mask = [1] * len(input_ids)

        # prepare for the teacher
        for _ in range(self.t_demo_batch_count):
            permutation_idxes = np.random.permutation(len(random_demo_idxes_list))[:self.t_demo_count + 1].tolist()
            random_demo_idxes = [random_demo_idxes_list[i] for i in permutation_idxes]
            if item in random_demo_idxes:
                random_demo_idxes.remove(item)
            else:
                random_demo_idxes = random_demo_idxes[:self.t_demo_count]
            t_demo_inputs_tokens_here, t_demo_count, t_demo_token_type_ids = self.demo_concate(
                random_demo_idxes, self.t_name, self.t_max_length, [], add_newlines=self.t_add_newlines
            )
            t_demo_inputs_tokens_list.append(t_demo_inputs_tokens_here)
        teacher_input_ids_list = []
        teacher_attention_mask_list = []
        teacher_token_type_ids_list = []
        teacher_demo_token_type_ids_list = []
        # ATTENTION: only choose one for teacher
        for t_demo_inputs_tokens_here in t_demo_inputs_tokens_list:
            teacher_input_tokens = [] if "gpt2" in self.t_name else [self.t_bos_token_id]
            teacher_input_tokens += t_demo_inputs_tokens_here
            teacher_demo_token_type_ids = [1] * len(teacher_input_tokens)

            teacher_input_tokens += dp[self.t_name + 'input_tokens']

            if len(teacher_input_tokens) + len(output_tokens) > self.t_max_length:
                teacher_input_tokens = teacher_input_tokens[
                                       len(teacher_input_tokens) + len(output_tokens) - self.t_max_length:]
                # add the seperated tokens for llama
                if self.t_add_newlines:
                    teacher_input_tokens += [self.t_sep_token_id]
            teacher_input_ids = teacher_input_tokens + output_tokens
            teacher_input_ids_list.append(teacher_input_ids)
            teacher_token_type_ids = [0] * len(teacher_input_tokens) + output_token_type_ids
            teacher_demo_token_type_ids += [0] * (len(teacher_token_type_ids) - len(teacher_demo_token_type_ids))
            teacher_token_type_ids_list.append(teacher_token_type_ids)
            teacher_demo_token_type_ids_list.append(teacher_demo_token_type_ids)
            teacher_attention_mask = [1] * len(teacher_input_ids)
            teacher_attention_mask_list.append(teacher_attention_mask)
            # demo_ids is designed for demonstration distillation model.

        outpout = {"input_ids": input_ids,
                   "attention_mask": attention_mask,
                   "token_type_ids": token_type_ids,
                   "teacher_input_ids": teacher_input_ids_list,
                   "teacher_attention_mask": teacher_attention_mask_list,
                   "teacher_token_type_ids": teacher_token_type_ids_list,
                   "demo_ids": demo_ids,
                   "demo_attention_mask": demo_attention_mask,
                   "demo_token_type_ids": s_demo_token_type_ids,
                   "task_idx": [task_idx],
                   "teacher_demo_token_type_ids": teacher_demo_token_type_ids_list
                   }

        return outpout

    def __getitem__(self, item):
        dp = self.target_dataset[item]
        task = dp['task']
        task_idx = self.task_idx_map.get(task, 0)
        all_task_demo_idxes = self.task_dict[task]
        is_classification = self.task_dict_is_clf[task]
        if "t5" in self.t_name.lower():
            return_dict = self.basic_t5_getitem(item, demo_count=self.demo_count)
        else:
            return_dict = self.basic_getitem(item, demo_count=self.demo_count)
        return_dict['is_classification'] = [is_classification]
        return return_dict


class C4PretrainDataset(Dataset):
    def __init__(self, data_args, tokenizer):
        super().__init__()
        # ATTENTION: The teacher and student are from the same family
        self.data_args = data_args
        catched_file = data_args.catched_file
        self.tokenizer_name = tokenizer.name_or_path.split("/")[-1]
        # ATTENTION: For llama and vicuna, the tokenizer should use slow mode
        if os.path.exists(catched_file) is False:
            dataset = load_dataset("bigheiniuJ/MyC4Validation", split='validation')
            token_fn = partial(self.tokenize_text, tokenizer=tokenizer)
            dataset = dataset.map(token_fn, batched=True, desc="Tokenization")
            dataset.save_to_disk(catched_file)
            self.dataset = dataset
        else:
            self.dataset = load_from_disk(catched_file)
        if type(self.dataset) is DatasetDict:
            self.dataset = self.dataset['validation']
        self.student_input_percent = data_args.student_input_percent
        self.num_tasks = 1
        self.task_idx_map = None
        self.shuffle_input_qm_prob = data_args.shuffle_input_qm_prob
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def tokenize_text(self, batch, tokenizer):
        input_tokens = tokenizer([i for i in batch['text']], add_special_tokens=False, truncation=True)['input_ids']
        batch['input_ids'] = input_tokens
        return batch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        teacher_input_ids = self.dataset[item]['input_ids']
        teacher_attention_mask = [1] * len(teacher_input_ids)
        s_input_len = int(self.student_input_percent * len(self.dataset[item]['input_ids']))
        demo_input_ids = teacher_input_ids[:s_input_len]
        input_ids = teacher_input_ids[s_input_len:]
        demo_attention_mask = [1] * len(demo_input_ids)
        attention_mask = [1] * len(input_ids)
        teacher_demo_token_type_ids = [1] * len(demo_input_ids) + [0] * len(input_ids)
        if self.data_args.split_query_pretrain_pro > 0:
            target_len = int(self.student_input_percent * len(input_ids))
            teacher_token_type_ids = [0] * (len(teacher_input_ids) - target_len) + [1] * target_len
        else:
            target_len = 0
            teacher_token_type_ids = teacher_attention_mask

        if "t5" in self.tokenizer_name:
            teacher_input_ids = demo_input_ids
            teacher_attention_mask = demo_attention_mask
            # just a placeholder
            teacher_token_type_ids = demo_attention_mask
            decoder_input_ids = [self.pad_token_id] + input_ids
            if self.data_args.split_query_pretrain_pro > 0:
                decoder_token_type_ids = [0] * (len(decoder_input_ids) - target_len) + [1] * target_len
            else:
                decoder_token_type_ids = [1] * len(decoder_input_ids)
            decoder_attention_mask = [1] * len(decoder_input_ids)

        is_consecutive = 1

        output = {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": attention_mask,
                "teacher_input_ids": teacher_input_ids,
                "teacher_attention_mask": teacher_attention_mask,
                "teacher_token_type_ids": teacher_token_type_ids,
                "demo_ids": demo_input_ids,
                "demo_attention_mask": demo_attention_mask,
                "teacher_demo_token_type_ids": teacher_demo_token_type_ids,
                "is_consecutive": [is_consecutive]
                }
        if "t5" in self.tokenizer_name:
            output['decoder_input_ids'] = decoder_input_ids
            output['decoder_token_type_ids'] = decoder_token_type_ids
            output['decoder_attention_mask'] = decoder_attention_mask
        return output




def expand_concate(small_list, large_list, repeat_count):
    expand_list = flatten([[i] * repeat_count for i in small_list])
    output_list = [i + j if type(i) is list else [i] + j for i, j in zip(expand_list, large_list)]

    if (len(output_list) == len(large_list) and len(small_list) * repeat_count == len(large_list)) is False:
        print((len(output_list), len(large_list), len(small_list), repeat_count))
        raise NotImplementedError
    return output_list

def repeat_list(small_list, repeat_count):
    expand_list = flatten([[i] * repeat_count for i in small_list])
    return expand_list

def data_collator_with_padding(batchs, t_tokenizer, is_fp16, is_left_pad=False):
    # TODO: effectiveness of left_pad
    # TODO: Handle text2text tasks
    # simply append 0s to all the tensors.
    new_batchs = defaultdict(list)
    for batch in batchs:
        for key, value in batch.items():
            if value is None:
                continue
            new_batchs[key].append(value)
    num_class = -1
    # for inference
    if "label_candidates" in batchs[0]:
        num_class = max([len(i['label_candidates']) for i in batchs])
        new_batchs['label_candidates'] = [i+[i[-1]] * (num_class-len(i)) for i in new_batchs['label_candidates']]
        # padding the options for qa relevant tasks.
        new_batchs['label_candidates_token_type_ids'] = [i+[i[-1]] * (num_class-len(i)) for i in new_batchs['label_candidates_token_type_ids']]
        new_batchs['label_candidates_attention_mask'] = [i+[i[-1]] * (num_class-len(i)) for i in new_batchs['label_candidates_attention_mask']]
        new_batchs['real_num_class_mask'] = [[1] * len(i['label_candidates']) + [0] * (num_class-len(i['label_candidates'])) for i in batchs]
        # padding the answer choices
        # handle  3D tensors  about label candidates
        # num_class = len(batchs[0]['label_candidates'])
        # flatten_them
        new_batchs['label_candidates'] = flatten(new_batchs['label_candidates'])
        new_batchs['label_candidates_token_type_ids'] = flatten(new_batchs['label_candidates_token_type_ids'])
        new_batchs['label_candidates_attention_mask'] = flatten(new_batchs['label_candidates_attention_mask'])
        # concatenate teacher input ids and candidates
        if "decoder_input_ids" in new_batchs.keys():
            new_batchs['teacher_input_ids'] = repeat_list(new_batchs['teacher_input_ids'], repeat_count=num_class)
            new_batchs['teacher_attention_mask'] = repeat_list(new_batchs['teacher_attention_mask'], repeat_count=num_class)
            new_batchs['input_ids'] = repeat_list(new_batchs['input_ids'], repeat_count=num_class)
            new_batchs['attention_mask'] = repeat_list(new_batchs['attention_mask'], repeat_count=num_class)
            new_batchs['token_type_ids'] = repeat_list(new_batchs['token_type_ids'], repeat_count=num_class)
            new_batchs['decoder_input_ids'] = expand_concate(new_batchs['decoder_input_ids'],
                                                             new_batchs['label_candidates'],
                                                             repeat_count=num_class
                                                             )
            new_batchs['decoder_attention_mask'] = expand_concate(new_batchs['decoder_attention_mask'],
                                                             new_batchs['label_candidates_attention_mask'],
                                                             repeat_count=num_class
                                                             )
            new_batchs['decoder_token_type_ids'] = expand_concate(new_batchs['decoder_token_type_ids'],
                                                                  new_batchs['label_candidates_token_type_ids'],
                                                                  repeat_count=num_class
                                                                  )

        else:
            new_batchs['teacher_input_ids'] = expand_concate(new_batchs['teacher_input_ids'],
                                                             new_batchs['label_candidates'],
                                                             repeat_count=num_class
                                                             )
            new_batchs['teacher_attention_mask'] = expand_concate(new_batchs['teacher_attention_mask'],
                                                                  new_batchs['label_candidates_attention_mask'],
                                                                  repeat_count=num_class)
            new_batchs['teacher_token_type_ids'] = expand_concate(new_batchs['teacher_token_type_ids'],
                                                                  new_batchs['label_candidates_token_type_ids'],
                                                                  repeat_count=num_class
                                                                  )

            new_batchs['input_ids'] = expand_concate(new_batchs['input_ids'],
                                                     new_batchs['label_candidates'],
                                                     repeat_count=num_class
                                                     )
            new_batchs['attention_mask'] = expand_concate(new_batchs['attention_mask'],
                                                          new_batchs['label_candidates_attention_mask'],
                                                          repeat_count=num_class)
            new_batchs['token_type_ids'] = expand_concate(new_batchs['token_type_ids'],
                                                          new_batchs['label_candidates_token_type_ids'],
                                                          repeat_count=num_class
                                                          )
        new_batchs['task_idx'] = flatten([[i] * num_class for i in new_batchs['task_idx']])

        # cancatenate input_ids and candidates

    # pad everything up
    padded_batches = {}
    for key, value in new_batchs.items():
        if len(value) == 0:
            continue
        if len(value[0]) > 0 and type(value[0][0]) is list:
            value = list(chain.from_iterable(value))
        cur_max_len = max([len(i) for i in value])
        if is_fp16 and "clf_label" != key:
            cur_max_len = int(math.ceil(cur_max_len / 8.0) * 8)
        padding_value = 0 if key != 'label' else -100
        try:
            if is_left_pad:
                value = torch.tensor([[padding_value] * (cur_max_len - len(i)) + i[:cur_max_len] for i in value])
            else:
                value = torch.tensor([i[:cur_max_len] + [padding_value] * (cur_max_len - len(i)) for i in value])
        except:
            print("*****KEY ERROR HERE {}".format((key, value)))
            exit(-1)
        # ATTENTION: reshape into 2d for 3d elements
        padded_batches[key] = value

    padded_batches['num_class'] = num_class
    return padded_batches


def pretrain_data_collator(batches, is_fp16, is_left_pad=False):
    new_batches = defaultdict(list)
    for batch in batches:
        for key, value in batch.items():
            if value is None:
                continue
            new_batches[key].append(value)

    padded_batches = {}
    for key, value in new_batches.items():
        if len(value[0]) > 0 and type(value[0][0]) is list:
            value = list(chain.from_iterable(value))
        cur_max_len = max([len(i) for i in value])
        if is_fp16:
            cur_max_len = int(math.ceil(cur_max_len / 8.0) * 8)
        if is_left_pad:
            value = torch.tensor([[0] * (cur_max_len - len(i)) + i[:cur_max_len] for i in value])
        else:
            value = torch.tensor([i[:cur_max_len] + [0] * (cur_max_len - len(i)) for i in value])
        padded_batches[key] = value
    return padded_batches



