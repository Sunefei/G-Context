# README

## PromptTuning

Prompt Tuning is a technique for fine-tuning pre-trained language models (such as GPT, BERT, etc.) by optimizing the input "prompts" to improve the model’s performance on specific tasks. Prompt Tuning does not update all the parameters of the model but instead learns a small, trainable prompt template that guides the model to produce more accurate outputs.

## Setup

All required packages are listed in `requirements.txt`. To install them in a new environment, follow these steps:

```bash
conda create -n PromptTuning python=3.9
conda activate PromptTuning
pip install -r requirements.txt
```

## Usage

To run the model, follow these steps:

1. Navigate to the `scripts` directory:

    ```bash
    cd scripts
    ```

2. Run the inference script:

    ```bash
    sh prompt-tuning.sh
    ```

3. To change the length of prompt embedding, modify the argument `virtual_demo_len` in `./scripts/prompt-tuning.sh`.

For all available model arguments, refer to `./distill_training_args.py`.

## Change Dataset

Currently, we support the following datasets used in our paper: `SST-5`, `HellaSwag`, `MRPC`, and `MNLI`. 

To change the dataset:

- Modify the `dataset_name` argument in `./scripts/prompt-tuning.sh` to `sst5`, `hellaswag`, `mrpc` and `mnli`.

To add a new dataset:
- Modify the `dataset_name` argument in `./scripts/prompt-tuning.sh`.
- Add the code segment for reading the dataset in `DistillDataset.__init__` function in `./src/dataset_distill.py`. An acceptable dataset format should include the following fields: `task` (consistent with `dataset_name`), `input`, `output`, `options` and `seed` (default is 100).

## Change Inference LLM

You can directly change the argument `t_name` in `./scripts/prompt-tuning.sh`. The `t_name` refers to the name of inference LLM.