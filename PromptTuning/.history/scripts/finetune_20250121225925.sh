#

cd ..
t_name=gpt2-large
s_name=gpt2
demo_count=16
demo_batch_count=1
virtual_demo_len=100
task=class_to_class
state_dict_path="/home/syf/MEND/output/model.safetensors"
output_dir='output_finetune'

# accelerate launch  --config_file accelerate_config.yaml main.py \
python main.py \
      --output_dir ${output_dir} \
      --catched_file catched_data/cached-s_${s_name}-t_${t_name}-${task}-full.torch \
      --eval_catched_file  catched_data/cached-s_${s_name}-t_${t_name}-${task}_full_eval100shot.torch \
      --seed 100 \
      --demo_count ${demo_count} \
      --s_model_name_or_path ${s_name} \
      --model_name_or_path ${t_name} \
      --t_max_length 1024 \
      --s_max_length 900 \
      --save_steps=2000 \
      --save_total_limit=4 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --report_to wandb \
      --label_names clf_label \
      --do_train \
      --do_predict \
      --task ${task} \
      --virtual_demo_len ${virtual_demo_len} \
      --wandb_project_name MetaICLV4 \
      --do_eval \
      --evaluation_strategy steps \
      --eval_steps 2000 \
      --max_steps 30000 \
      --is_100_shot \
      --demo_batch_count ${demo_batch_count} \
      --is_fid \
      --learning_rate 1e-5 \
      --overwrite_output_dir \
      --load_best_model_at_end \
      --metric_for_best_model FinalFinal \
      --is_init_prompt_weight \
      --virtual_demo_init vocab \
      --is_query_kl_loss \
      --s_state_dict_path ${state_dict_path} \
