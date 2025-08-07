CUDA_VISIBLE_DEVICES=0 python "/home/arinjay/model-merging-adapters/finetune.py" \
    --base_model /home/models/Qwen1.5-4B \
    --task_name ARC-Challenge \
    --task_type LLM \
    --task_niche Commonsense \
    --cutoff_len 256 \
    --val_set_size 80 \
    --output_model "./models/unmerged/Qwen1.5-4B/${task_name}/"\
    --seed 42 \
    --batch_size 16 \
    --micro_batch_size 1 \
    --num_epochs 5 \
    --learning_rate 3e-4 \
    --eval_step 100 \
    --save_step 100 \
    --adapter_type LoRA \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_dora False \
    --use_rslora False \
    --group_by_length False \


