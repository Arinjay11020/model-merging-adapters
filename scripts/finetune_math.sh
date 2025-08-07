CUDA_VISIBLE_DEVICES=3 python /home/arinjay/model-merging-adapters/finetune_gsm8k.py   \
--base_model '/home/models/Meta-Llama-3.2-3B-Instruct'   \
--data_path 'gsm8k'   \
--output_dir '/home/arinjay/models/expert/math/lora/llama-3.2-3B-gsm8k'  \
--method lora \
--batch_size 8 \
--micro_batch_size 4 \
--num_epochs 3 \
--lora_r 32 \
--learning_rate 1e-5 \
--cutoff_len 256 \
--val_set_size 120 \
--apply_peft \
--eval_step 120 \
--save_step 120 \
--lora_alpha 64 \
--target_modules '["q_proj", "k_proj", "v_proj"]' \

