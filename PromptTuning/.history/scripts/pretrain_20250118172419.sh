cd ..
s_name=gpt2
t_name=gpt2-large
task=class_to_class
demo_count=16
demo_batch_count=1
bsz=1
lr=5e-5
output_dir='/path/to/output_dir'

accelerate  launch  --config_file accelerate_config.yaml main.py \
  --output_dir ${output_dir} \
  --catched_file catched_data/cached-c4-${t_name}.torch \
  --eval_catched_file catched_data/cached-s_${s_name}-t_${t_name}_${task}_full_eval-100shot.torch \
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
  --do_train \
  --task ${task} \
  --eval_steps 2000 \
  --evaluation_strategy steps \
  --virtual_demo_len 100 \
  --wandb_project_name MetaICLV4 \
  --max_steps 30000 \
  --is_100_shot \
  --demo_batch_count ${demo_batch_count} \
  --overwrite_output_dir \
  --is_c4_pretrain \
  --student_input_percent 0.8 \
  --is_init_prompt_weight \
  --learning_rate ${lr} \
  --do_eval \
  --do_predict \
  --is_only_auxiliary_loss \
  --is_query_kl_loss \
  --load_best_model_at_end \
  --metric_for_best_model FinalFinal \
  --temperature 1 \
  --is_no_gradient_check