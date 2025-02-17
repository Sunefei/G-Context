# README

## MEND: Meta dEmonstratioN Distillation for Efficient and Effective In-Context Learning

This repository is mainly based on the source code for MEND, which is a demonstration distillation method proposed in paper [MEND: Meta dEmonstratioN Distillation for Efficient and Effective In-Context Learning](https://arxiv.org/pdf/2403.06914). 

To adapt MEND to our specific requirements, we have modified portions of the original source code from the MEND repository. If you want to check the original version, please refer to [MEND](https://github.com/bigheiniu/MEND).

## Setup

All required packages can be found in `requirements.txt`. You can install them in a new environment with:

```bash
conda create -n MEND python=3.9
conda activate MEND
pip install -r requirements.txt
```

## Usage

### Pretrain

The orginal code of MEND utilize the validation set of C4 dataset for pretrain. You can directly load this dataset from the `/catched_data/cached-c4-gpt2-large.torch` and perform pretrain of follow the instructions in [MEND](https://github.com/bigheiniu/MEND). Or you can directly you our pretrain checkpoint in `output/pytorch_model.bin`

The original MEND code uses the validation set of the C4 dataset for pretraining. You can directly load this dataset from `/catched_data/cached-c4-gpt2-large.torch` follow the pretraining instructions in the [MEND repository](https://github.com/bigheiniu/MEND). Alternatively, you can use our pre-trained checkpoint available at `output/pytorch_model.bin` for reproduction of our experiment result in fine-tuning.

### Fine-Tuning

To fine-tune the model, follow these steps:

1. Navigate to the `scripts` directory:

    ```bash
    cd scripts
    ```

2. Run the fine-tuning script:

    ```bash
    sh finetune.sh
    ```

For all available model arguments, refer to `./distill_training_args.py`.



## Modules

Inside [src](./src) directory, you will find:

- [dataset_distill.py](./src/dataset_distill.py) - This houses both the pretrain C4 dataset class and the meta-train/meta-test dataset class.
- [model_distill.py](./src/model_distill.py)- This manages the interaction between the large language model and the context distillation model.
- [SmallModel.py](./src/SmallModel.py)- This file contains the implementation of the context distillation model.


## Change Dataset

Currently, we support the following datasets used in our paper: `sst5`, `Hellaswag`, `mrpc`, and `mnli`. Additionally, the original `metal` ICL dataset is supported.

To change the dataset:

1. Modify the `dataset_name` argument in `./scripts/finetune.sh`.

2. Update the `./config/class_to_class.json` file. Specifically, write out the dataset you want to use for training and evaluation. Using different datasets is allowed, and we test our model's ability to transfer to different tasks.


To switch back to the metaICL task, you can find the original JSON file in the [MEND repository](https://github.com/bigheiniu/MEND).


## Change Inference LLM

You can directly change the argument in `t_name` in `./scripts/finetune.sh`. The `t_name` refers to the teaching model name, which is used for training the student model and downstream inference. 

Note that you don't need to change the `s_name` argument, as it refers to the backbone model of MEND. For more information, you can refer to the [MEND paper](https://arxiv.org/pdf/2403.06914).


## Citation

If you use this code for your research, please cite MEND paper:

```
@inproceedings{
li2024mend,
title={{MEND}: Meta Demonstration Distillation for Efficient and Effective In-Context Learning},
author={Yichuan Li and Xiyao Ma and Sixing Lu and Kyumin Lee and Xiaohu Liu and Chenlei Guo},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=2Y5kBPtU0o}
}
```