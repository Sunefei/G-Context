#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=4,5

gpu=2
method=epr
num_ice=50
port=5324

model_name=gpt2-large
n_tokens=700
scr_batch_size=128
inf_batch_size=48

#model_name=EleutherAI/gpt-neo-2.7B
#n_tokens=1600
#scr_batch_size=8
#inf_batch_size=8

for task_name in mrpc
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}
#index_data=index_data/mrpc/index_dataset.json  
#retrieve_file=output/epr/mrpc/gpt2-large/retrieved.json
#scoered_file=output/epr/mrpc/gpt2-large/scored.json
  retrieve_file=${run_dir}/retrieved.json
  scored_file=${run_dir}/scored.json
  run_name=bert-fix_ctx-shared-bs64
  run_dir=${run_dir}/${run_name}
#utput/epr/mrpc/gpt2-large/
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      model_config.ctx_model_name=null  # share ctx model with q model



done


