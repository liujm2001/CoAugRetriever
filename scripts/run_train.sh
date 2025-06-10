# set -x
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

### parameters
dataset_name='nfcorpus'
model_name='Qwen2.5-7B-Instruct'
use_dense='False'
posk=0.2
negk=0.1
train_epochs=5
###

data_path='./dataset'
ckpt_path='./ckpt'
config_name="${model_name}-${train_epochs}epoch-${use_dense}dense-${posk}posk-${negk}negk"
micro_batch_size=16

python -m verl.trainer.main_ppo \
    reward_model.rule_based.use_dense=${use_dense} \
    reward_model.rule_based.posk=${posk} \
    reward_model.rule_based.negk=${negk} \
    reward_model.rule_based.embedmodel_path=${ckpt_path}/bge-base-en-v1.5 \
    algorithm.adv_estimator=grpo \
    data.queries_files=${data_path}/${dataset_name}/queries.jsonl \
    data.corpus_files=${data_path}/${dataset_name}/corpus.jsonl \
    data.train_files=${data_path}/${dataset_name}/qrels/train.tsv \
    data.test_files=${data_path}/${dataset_name}/qrels/test.tsv \
    data.prompt_q_files=${data_path}/prompt_q \
    data.prompt_k_files=${data_path}/prompt_k \
    data.train_batch_size=512 \
    data.val_batch_size=512 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    actor_rollout_ref.model.path=${ckpt_path}/${model_name} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=${micro_batch_size} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.2 \
    actor_rollout_ref.rollout.repetition_penalty=1.2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=${micro_batch_size} \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_before_train=False \
    +trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.project_name=RL4IR \
    trainer.experiment_name=${dataset_name}-${config_name} \
    trainer.default_local_dir=${ckpt_path}/${dataset_name}-${config_name}/ \
    trainer.total_epochs=${train_epochs} 2>&1 | tee ${ckpt_path}/log/verl_demo.log \
    # actor_rollout_ref.ref.fsdp_config.param_offload=True \
