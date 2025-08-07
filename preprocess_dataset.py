from datasets import load_dataset
def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True, base_model=None):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=add_eos_token  # Let the tokenizer handle special tokens
    )
    if add_eos_token and result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len:
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in (base_model or ""):  # Handle case where base_model might be None
            result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    if "chatglm" in (base_model or ""):
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result
def preprocess_dataset(task_name,task_type,concat_train,task_modality):#task_type=[SEQ_CLS,SEQ_2_SEQ_LM,CAUSAL_LM,TOKEN_CLS,QUESTION_ANS,FEATURE_EXTRACTION]
    pass
def generate_and_tokenize_prompt(data_point):
    full_prompt=generate_prompt(data_point)
    tokenized_full_prompt=tokenize(full_prompt,tokenizer,cutoff_len,add_eos_token=False)
    user_prompt_len=len(tokenized_full_prompt["input_ids"])
    tokenized_full_prompt["labels"] = [
                                                  -100
                                              ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                    user_prompt_len:
                                                                    ]  # could be sped up, probably
    return tokenized_full_prompt
def generate_prompt(data_point):
    instruction = data_point.get("instruction", "")
    input_text = data_point.get("input", "")
    output_text = data_point.get("output", "")
    if input_text:
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output_text}"""
    else:
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output_text}"""
    return prompt
def format_gsm8k_with_apply_chat_template(example):
        question = example["question"].strip()
        answer = example["answer"].strip()

        q_template ='Given the following problem, reason and give a final answer to the problem.\nProblem: {quest}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
        cleanup_pattern = re.compile(r'\$?<<.*?>>')
        parts = answer.split("####")
        raw_reasoning = parts[0].strip()
        final_ans = parts[1].strip() if len(parts) > 1 else ""

        reasoning = cleanup_pattern.sub("", raw_reasoning)
        reasoning = reasoning.replace("\n", " ").strip()

        usr = (
            "Given the following problem, reason and give a final answer to the problem.\n"
            f"Problem: {question}\n"
            'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.\n\n'
            
        )
        asst = (f"{reasoning}"
                f"The final answer is {final_ans}\n")
        
        conversation = [
            # {"role": "system", "content": "You are a helpful assistant that's good at solving math problems step by step."},
            # {"role":"user", "content":""},
            {"role": "user", "content": usr},
            {"role": "assistant", "content": asst}
        ]
        
        # Apply the model's chat template
        formatted_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}