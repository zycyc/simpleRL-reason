
HDFS_HOME=/data/simpleRL-reason/train
RUN_NAME=Qwen2.5-Math-7B_ppo_from_base_math_lv35_quicker_experiment

TOKENIZERS_PARALLELISM=True python3 openrlhf/cli/train_ppo_ray_box.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 0 \
    --reward_num_gpus_per_node 0 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --vllm_num_engines 4 \
    --vllm_tensor_parallel_size 1 \
    --colocate_actor_ref \
    --pretrain $HDFS_HOME/model_hub/models--Qwen--Qwen2.5-Math-7B/ \
    --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
    --micro_train_batch_size 8 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 16 \
    --rollout_batch_size 1024 \
    --temperature 0.6 \
    --n_samples_per_prompt 1 \
    --max_samples 100000 \
    --max_epochs 1 \
    --num_episodes 20 \
    --prompt_max_len 1024 \
    --generate_max_len 3000 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data  data/math_level3to5_data_processed_with_qwen_prompt.json \
    --input_key input \
    --normalize_reward \
    --flash_attn \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --save_steps 4 \
    --load_checkpoint \
    --use_wandb 5fb2c3eb35cb3bc0124a02069ce91eedc6570e5a \
    --wandb_run_name $RUN_NAME \
    --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME  \
    --max_ckpt_num 20000