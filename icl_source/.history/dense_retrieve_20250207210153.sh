#!/bin/bash

export WANDB_PROJECT=graphICL  # change if needed
export WANDB_ENTITY=  yuxiang_23_university_of_illinois_urbana_champaign_org # change to your wandb account
export WANDB_API_KEY= c372f8378f45ad2d2d6839b40dcc28f1aaf4b58f # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

gpu=2
method=epr
num_ice=50
port=5324

model_name=gpt2-large
n_tokens=700
scr_batch_size=128
inf_batch_size=48
task_name=swag
#model_name=EleutherAI/gpt-neo-2.7B
#n_tokens=1600
#scr_batch_size=8
#inf_batch_size=8

#run_dir = output/epr/swag/gpt2-large

for task_name in swag
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}


#retrieved_file = output/epr/swag/gpt2-large/retrieved.json
#index_data=index_data/swag/index_dataset.json
  retrieve_file=${run_dir}/retrieved.json
#scored_file=output/epr/swag/gpt2-large/scored.json
  scored_file=${run_dir}/scored.json


#run_dir=output/epr/swag/gpt2-large/bert-fix_ctx-shared-bs64
  run_name=bert-fix_ctx-shared-bs64
  run_dir=${run_dir}/${run_name}



#retrieve_file = output/epr/swag/gpt2-large/bert-fix_ctx-shared-bs64/train_retrieved.json
  retrieve_file=${run_dir}/train_retrieved.json
  python dense_retriever.py \
      hydra.run.dir=${run_dir}/dense_retriever \
      output_file=${retrieve_file} \
      num_ice=${num_ice} \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      pretrained_model_path=${run_dir} \
      faiss_index=${run_dir}/index

done

