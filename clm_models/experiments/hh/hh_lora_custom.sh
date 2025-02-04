deepspeed ../train.py \
  --model_name_or_path PygmalionAI/pygmalion-6b \
  --tokenizer_name PygmalionAI/pygmalion-6b \
  --dataset_name AlekseyKorshuk/gpteacher-role-play-chatml \
  --train_to_probs False \
  --do_train \
  --do_eval \
  --logging_strategy steps \
  --evaluation_strategy epoch \
  --eval_steps 1 \
  --save_strategy epoch \
  --save_steps 1 \
  --logging_steps 100 \
  --logging_first_step \
  --report_to all \
  --output_dir ./checkpoints/gptj_hh_lora \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing False \
  --max_eval_samples 500 \
  --num_train_epochs 10 \
  --eval_first_step False \
  --learning_rate 1e-5 \
  --lr_scheduler_type "cosine" \
  --fp16 \
  --seed 99 \
  --validation_split_percentage 1 \
  --remove_unused_columns False \
  --deepspeed ./deepspeed_configs/ds_config_stage_3.json \
  --clean_enabled False \
  --block_size 512 \
  --use_lora True \
  --warmup_ratio 0.1 \
  --weight_decay 0.00001 \
  --push_to_hub True \
  --hub_model_id "hedronstone/6b-gpteacher-role-play-chatml" \
  --hub_strategy end \
