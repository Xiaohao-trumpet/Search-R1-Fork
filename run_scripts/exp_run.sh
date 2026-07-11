#!/bin/bash
# Parametrized Search-R1 improvement experiment runner.
# Variant knobs via env vars (all optional; defaults reproduce the binary-EM + vanilla-GRPO baseline):
#   VARIANT           : name/tag for logs+ckpt        (default: baseline)
#   REWARD_TYPE       : em | f1_shaped                 (default: em)
#   NORM_ADV_BY_STD   : True | False (False=Dr.GRPO)   (default: True)
#   CLIP_HIGH         : upper clip (e.g. 0.28) | null  (default: null)
#   CLIP_LOW          : lower clip | null              (default: null)
#   ENTROPY_COEF      : entropy bonus                  (default: 0.001)
#   DYN_SAMPLING      : True | False                   (default: False)
#   GPUS              : e.g. "4,5"                      (default: 4,5)
#   STEPS             : total training steps           (default: 20)
#   NAGENT            : GRPO group size                (default: 5)
#   TRAIN_BS          : train batch size (prompts)     (default: 8)
set -u
source /data/mambaforge/etc/profile.d/conda.sh
conda activate searchr1
cd /mnt/backup1/lgc/exp/Search-R1

VARIANT=${VARIANT:-baseline}
REWARD_TYPE=${REWARD_TYPE:-em}
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-True}
CLIP_HIGH=${CLIP_HIGH:-null}
CLIP_LOW=${CLIP_LOW:-null}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
DYN_SAMPLING=${DYN_SAMPLING:-False}
GPUS=${GPUS:-4,5}
STEPS=${STEPS:-20}
NAGENT=${NAGENT:-5}
TRAIN_BS=${TRAIN_BS:-8}
VAL_NUM=${VAL_NUM:-8}
TESTFREQ=${TESTFREQ:-10}

NGPU=$(echo $GPUS | tr ',' '\n' | wc -l)
LOGDIR=/mnt/backup1/lgc/search-r1-data/exp_logs
mkdir -p $LOGDIR
LOG=$LOGDIR/${VARIANT}.log

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$GPUS
export DATA_DIR=/mnt/backup1/lgc/search-r1-data/nq_search
# 1.5B is format-competent untrained (unlike 0.5B) -> base policy produces scorable
# rollouts, so reward becomes non-zero and H1/H4 are measurable. Overridable via MODEL.
export BASE_MODEL=${MODEL:-/data/share/qwen/Qwen2.5-1.5B-Instruct}
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[exp_run] VARIANT=$VARIANT REWARD_TYPE=$REWARD_TYPE NORM_ADV_BY_STD=$NORM_ADV_BY_STD CLIP=[${CLIP_LOW},${CLIP_HIGH}] ENT=$ENTROPY_COEF DYN=$DYN_SAMPLING GPUS=$GPUS STEPS=$STEPS NAGENT=$NAGENT TRAIN_BS=$TRAIN_BS" | tee $LOG

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=$VAL_NUM \
    data.train_batch_size=$TRAIN_BS \
    data.val_batch_size=8 \
    data.max_prompt_length=1536 \
    data.max_response_length=256 \
    data.max_start_length=768 \
    data.max_obs_length=256 \
    data.shuffle_train_dataloader=True \
    data.reward_type=$REWARD_TYPE \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std=$NORM_ADV_BY_STD \
    algorithm.dynamic_sampling=$DYN_SAMPLING \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BS \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.n_agent=$NAGENT \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger=['console'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TESTFREQ \
    trainer.project_name=Search-R1-improve \
    trainer.experiment_name=$VARIANT \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$STEPS \
    trainer.default_local_dir=/mnt/backup1/lgc/search-r1-data/ckpt/$VARIANT \
    max_turns=3 \
    do_search=true \
    retriever.url="http://127.0.0.1:8002/retrieve" \
    retriever.topk=3 2>&1 | tee -a $LOG
echo "[exp_run] DONE VARIANT=$VARIANT exit=${PIPESTATUS[0]}" | tee -a $LOG
