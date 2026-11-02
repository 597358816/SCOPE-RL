set -x

MODEL_PATH=Qwen/Qwen3-4B  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=./train_dapo.parquet \
    data.val_files=./train_dapo.parquet \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.n=8 \
    worker.rollout.gpu_memory_utilization=0.6 \
    algorithm.disable_kl=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    trainer.experiment_name=qwen3-4b-SCOPE \
    trainer.algorithm=SCOPE \
    trainer.entropy_base=0.5 \
    trainer.positive_sample_num=16 \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=2 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=20 \
    trainer.save_limit=1

