export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=8000 

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train true \
    --use_fast_tokenizer \
    --flash_attn auto\
    --model_name_or_path \
    --dataset openreview \
    --template \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir \
    --overwrite_cache true \
    --overwrite_output_dir true \
    --warmup_steps 10 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 180000000 \
    --learning_rate 3e-4 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 100 \
    --plot_loss true \
    --num_train_epochs 1 \
    --bf16 \
    --ddp_find_unused_parameters false \
    --val_size 0.1 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --per_device_eval_batch_size 1