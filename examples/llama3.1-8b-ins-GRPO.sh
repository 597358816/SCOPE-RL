set -x

MODEL_PATH=meta-llama/Llama-3.1-8B-Instruct  # replace it with your local file path

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=config.yaml \
    data.train_files=hiyouga/math12k@train \
    data.val_files=hiyouga/math12k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    data.max_response_length=8192 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.n=8 \
    algorithm.disable_kl=true \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=llama3.1-8b-ins-GRPO \
    trainer.algorithm=GRPO \
    trainer.n_gpus_per_node=8 \
    trainer.total_episodes=4

