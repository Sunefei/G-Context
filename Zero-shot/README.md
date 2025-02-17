# README

## Zero-shot

Zero-shot learning is a machine learning method that enables a model to perform inference and make predictions on tasks or categories it has never been directly trained on. In simple terms, zero-shot methods allow a model to make effective judgments about unseen categories or tasks by leveraging its existing knowledge.

## Setup

All required packages are listed in `requirements.txt`. To install them in a new environment, follow these steps:

```bash
conda create -n ZeroShot python=3.9
conda activate ZeroShot
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
    sh zero-shot.sh
    ```

For all available model arguments, refer to `./distill_training_args.py`.

## Change Dataset

Currently, we support the following datasets used in our paper: `SST-5`, `HellaSwag`, `MRPC`, and `MNLI`. 

To change the dataset:

- Modify the `dataset_name` argument in `./scripts/zero-shot.sh` to `sst5`, `hellaswag`, `mrpc` and `mnli`.

To add a new dataset:
- Modify the `dataset_name` argument in `./scripts/zero-shot.sh`.
- Add the code segment for reading the dataset in `DistillDataset.__init__` function in `./src/dataset_distill.py`. An acceptable dataset format should include the following fields: `task` (consistent with `dataset_name`), `input`, `output`, `options` and `seed` (default is 100).

## Change Inference LLM

You can directly change the argument `t_name` in `./scripts/zero-shot.sh`. The `t_name` refers to the name of inference LLM.