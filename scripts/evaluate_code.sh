 CUDA_VISIBLE_DEVICES=2 lm_eval \
    --model hf \
    --model_args pretrained="/home/models/Meta-Llama-3.2-3B-Instruct",peft="/home/arinjay/model-merging-adapters/models/expert/code/lora/llama-3.2-3B-code/final" \
    --tasks humaneval_instruct \
    --log_samples \
    --output_path ./result/code-ft/llama/code-ft/${METHOD_NAME}/llama3.2-3b-instruct-code-${METHOD_NAME}-${LEARNING_RATE}  \
    --batch_size 32   --trust_remote_code --confirm_run_unsafe_code