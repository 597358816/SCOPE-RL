set -x

MODEL_PATH=Qwen/Qwen2.5-7B  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=./train_dapo.parquet \
    data.val_files=./train_dapo.parquet \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.max_response_length=2048 \
    worker.rollout.n=8 \
    algorithm.disable_kl=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.micro_batch_size_per_device_for_update=8 \
    worker.actor.micro_batch_size_per_device_for_experience=16 \
    trainer.experiment_name=qwen2.5-7b-GRPO \
    trainer.algorithm=GRPO \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=6 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=20 \
    trainer.save_limit=8

