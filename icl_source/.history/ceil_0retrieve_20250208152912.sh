#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

gpu=2
method=dpp-epr-random
num_ice=50
port=9927

model_name=gpt2-large
n_tokens=700
scr_batch_size=128
inf_batch_size=48
