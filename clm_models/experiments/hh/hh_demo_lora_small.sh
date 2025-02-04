deepspeed train.py \
  --model_name_or_path EleutherAI/gpt-j-6b \
  --tokenizer_name EleutherAI/gpt-j-6b \
  --dataset_name AlekseyKorshuk/hh-lmgym-demo \
  --train_to_probs False \
  --do_train \
  --logging_strategy steps \
  --evaluation_strategy epoch \
  --eval_steps 1 \
  --save_strategy epoch \
  --save_steps 1 \
  --logging_steps 100 \
  --logging_first_step \
  --report_to all \
  --output_dir ./checkpoints/gptj_hh_lora_test \
  --overwrite_output_dir \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing False \
  --max_eval_samples 64 \
  --num_train_epochs 1 \
  --eval_first_step False \
  --learning_rate 1e-5 \
  --lr_scheduler_type "cosine" \
  --fp16 \
  --seed 99 \
  --validation_split_percentage 1 \
  --remove_unused_columns False \
  --deepspeed ./deepspeed_configs/ds_config_stage_2_offload.json \
  --clean_enabled False \
  --block_size 512 \
  --use_lora True \
  --warmup_ratio 0.03 \
  --weight_decay 0.00001 \
  --max_train_samples 64 \
  --push_to_hub True \
  --hub_model_id "AlekseyKorshuk/lora-save-test-6b" \
  --hub_strategy end