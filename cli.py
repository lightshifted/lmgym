import click
import subprocess

@click.command()
@click.help_option('-h', '--help', help='Display help information about this command.')
@click.option("-m""--model_name_or_path", default="PygmalionAI/pygmalion-6b", help="Path or name of the model to use.")
@click.option("--tokenizer_name", default="PygmalionAI/pygmalion-6b", help="Name of the tokenizer to use.")
@click.option("--dataset_name", default="AlekseyKorshuk/gpteacher-role-play-chatml", help="Name of the dataset to use.")
@click.option("--train_to_probs", default=False, type=bool, help="If set to True, the model will be trained to probabilities.")
@click.option("--do_train", is_flag=True, default=False, help="Set this flag to enable training.")
@click.option("--do_eval", is_flag=True, default=False, help="Set this flag to enable evaluation.")
@click.option("--logging_strategy", default="steps", help="The logging strategy to use. Can be 'steps' or 'epoch'.")
@click.option("--evaluation_strategy", default="epoch", help="The evaluation strategy to use. Can be 'steps' or 'epoch'.")
@click.option("--eval_steps", default=1, type=int, help="The number of steps between evaluations.")
@click.option("--save_strategy", default="epoch", help="The saving strategy to use. Can be 'steps' or 'epoch'.")
@click.option("--save_steps", default=1, type=int, help="The number of steps between saving the model.")
@click.option("--logging_steps", default=100, type=int, help="The number of steps between logging.")
@click.option("--logging_first_step", is_flag=True, default=False, help="Set this flag to enable logging at the first step.")
@click.option("--report_to", default="all", help="Where to report the results. Can be 'all', 'tensorboard', 'wandb', etc.")
@click.option("--output_dir", default="./checkpoints/gptj_hh_lora", help="The directory where the model will be saved.")
@click.option("--overwrite_output_dir", is_flag=True, default=False, help="Set this flag to overwrite the output directory.")
@click.option("--per_device_train_batch_size", default=4, type=int, help="The batch size per device during training.")
@click.option("--gradient_accumulation_steps", default=2, type=int, help="The number of steps for gradient accumulation.")
@click.option("--gradient_checkpointing", default=False, type=bool, help="If set to True, the model will use gradient checkpointing.")
@click.option("--max_eval_samples", default=500, type=int, help="The maximum number of evaluation samples.")
@click.option("--num_train_epochs", default=5, type=int, help="The number of training epochs.")
@click.option("--eval_first_step", default=False, type=bool, help="If set to True, the model will be evaluated at the first step.")
@click.option("--learning_rate", default=1e-5, type=float, help="The learning rate to use during training.")
@click.option("--lr_scheduler_type", default="cosine", help="The type of learning rate scheduler to use.")
@click.option("--fp16", is_flag=True, default=False, help="Set this flag to enable FP16 training.")
@click.option("--seed", default=99, type=int, help="The seed for initializing training.")
@click.option("--validation_split_percentage", default=1, type=int, help="The percentage of the data to use for validation.")
@click.option("--remove_unused_columns", default=False, type=bool, help="If set to True, unused columns in the dataset will be removed")
@click.option("--remove_unused_columns", default=False, type=bool, help="If set to True, unused columns in the dataset will be removed.")
@click.option("--deepspeed", default="./deepspeed_configs/ds_config_stage_3.json", help="Path to the deepspeed configuration file.")
@click.option("--clean_enabled", default=False, type=bool, help="If set to True, cleaning will be enabled.")
@click.option("--block_size", default=512, type=int, help="The block size for the model.")
@click.option("--use_lora", default=True, type=bool, help="If set to True, the model will use LoRA.")
@click.option("--warmup_ratio", default=0.03, type=float, help="The warmup ratio to use during training.")
@click.option("--weight_decay", default=0.00001, type=float, help="The weight decay to use during training.")
@click.option("--push_to_hub", default=True, type=bool, help="If set to True, the model will be pushed to the Hugging Face Model Hub.")
@click.option("--hub_model_id", default="hedronstone/6b-gpteacher-role-play-chatml", help="The ID of the model on the Hugging Face Model Hub.")
@click.option("--hub_strategy", default="end", help="The strategy to use for pushing to the Hugging Face Model Hub.")

def main(**kwargs):
    """
    This CLI is used for deepspeed training. For each argument, provide a meaningful value 
    that aligns with the training requirement. If you're unsure about any argument, use --help
    to learn more about it.
    """
    cmd = ["deepspeed", "./train.py"]
    
    for k, v in kwargs.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k} {v}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred during the execution of the deepspeed training. Error details: {e}", err=True)

if __name__ == "__main__":
    main()

