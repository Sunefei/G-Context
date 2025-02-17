#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=2
method=epr
num_ice=50
port=5324

model_name=gpt2-large
n_tokens=700
scr_batch_size=128
inf_batch_size=48

for task_name in mrpc
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  retrieve_file=${run_dir}/retrieved.json
  python bm25_retriever.py \
      hydra.run.dir=${run_dir}/bm25_retriever \
      output_file=${retrieve_file} \
      num_candidates=50 \
      num_ice=1 \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      dataset_split=train \
      ds_size=44000 \
      query_field=a \
      index_reader.field=a

done