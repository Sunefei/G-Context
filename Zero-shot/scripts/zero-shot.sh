cd ..
s_name=gpt2
t_name=gpt2-large
dataset_name=hellaswag
task=class_to_class
demo_count=0
demo_batch_count=1
bsz=1
lr=5e-5
output_dir='output'

python main.py \
  --output_dir ${output_dir} \
  --dataset_name ${dataset_name} \
  --catched_file catched_data/cached-${dataset_name}-${t_name}.torch \
  --eval_catched_file catched_data/cached-${dataset_name}-${t_name}_eval.torch \
  --seed 100 \
  --demo_count ${demo_count} \
  --s_model_name_or_path ${s_name} \
  --model_name_or_path ${t_name} \
  --t_max_length 1024 \
  --s_max_length 900 \
  --save_steps=2000 \
  --save_total_limit=4 \
  --per_device_train_batch_size ${bsz} \
  --per_device_eval_batch_size 1 \
  --report_to wandb \
  --label_names clf_label \
  --task ${task} \
  --virtual_demo_len 100 \
  --wandb_project_name MetaICLV4 \
  --max_steps 10000 \
  --is_100_shot \
  --demo_batch_count ${demo_batch_count} \
  --overwrite_output_dir \
  --student_input_percent 0.8 \
  --is_init_prompt_weight \
  --learning_rate ${lr} \
  --do_predict \
  --is_only_auxiliary_loss \
  --is_query_kl_loss \
  --metric_for_best_model FinalFinal \
  --temperature 1 \
  --is_no_gradient_check