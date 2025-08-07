CUDA_VISIBLE_DEVICES=3 python /home/arinjay/model-merging-adapters/finetune_code.py   \
--base_model '/home/models/Meta-Llama-3.2-3B-Instruct'   \
--data_path '/home/arinjay/model-merging-adapters/dataset/LLMs/Code/code_python_38k.json'   \
--output_dir '/home/arinjay/models/expert/code/lora/llama-3.2-3B-code'  \
--method lora \
--batch_size 8 \
--micro_batch_size 4 \
--num_epochs 3 \
--num_epochs_coop 3 \
--lora_r 32 \
--learning_rate 1e-5 \
--cutoff_len 256 \
--val_set_size 120 \
--apply_peft \
--eval_step 2000 \
--save_step 2000 \
--lora_alpha 64 \
--posthoc_app 0 \
--target_modules '["q_proj", "k_proj", "v_proj"]' \

