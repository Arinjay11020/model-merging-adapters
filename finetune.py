#Right now trying to add only a few finetuning ways, might be tailored to LLM only
import json
import os
import sys
import fire
import random
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, set_seed 
from datasets import load_dataset
import typing
from typing import List, Optional, Union, ClassVar
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    TaskType,
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification
)
from hf_argparser import HfArgumentParser
#from peft.src.peft.tuners.tuners_utils import BaseTunerLayer
from peft import LoraConfig,LoftQConfig,LoraRuntimeConfig,LoraModel
from peft import AdaLoraConfig
from peft import PrefixTuningConfig
from dataclasses import dataclass,field
from preprocess_dataset import preprocess_dataset,tokenize,generate_and_tokenize_prompt,generate_prompt
@dataclass
class DataArguments:
    task_name:Optional[str]=field(
        default=None,
        metadata={"help":"The name of the dataset"}
    )
    task_niche:str=field(
        default="Commonsense",
        metadata={"help":"Decide type of tasks"}
    )
    concat_train:Optional[bool]=field(
        default=False,
        metadata={"help":"In case you need to concat train split of many datasets together"}
    )
    task_type:Optional[str]=field(
        default="LLM",
        metadata={"help":"Task should be suited for LLMs, MLLMs, etc"}
    )
    cutoff_len: int=field(
        default=256,
        metadata={"help":"Max length for tokenizer"}
    )
    val_set_size: int=field(
        default=120,
        metadata={"help":"Validation samples, set to 0 for training on entire dataset"}
    )
@dataclass
class ModelArguments:
    base_model: str=field(
        default="/home/models/llama-7b-hf",
        metadata={"help":"Path to the base model"}
    )
    output_model: Optional[str]=field(
        default="./models/LLMs/model_name",
        metadata={"help":"Path to save finetuned weights"}
    )
@dataclass
class TrainingArguments:
    seed:int=field(
        default=42,
        metadata={"help":"sets seeds for numpy, python, transformers, cuda, pytorch, and environment"}
    )
    batch_size: int=field(
        default=16,
        metadata={"help":"the batch size"}
    )
    micro_batch_size: int=field(
        default=1,
        metadata={"help":"divide batch size with this for gradient accumulation step"}
    )
    num_epochs: int=field(
        default=3,
        metadata={"help":"Number of epochs"}
    )
    learning_rate: float=field(
        default=3e-4,
        metadata={"help":"learning rate"}
    )
    eval_step: int=field(
        default=120,
        metadata={"help":"Evaluation steps"}
    )
    save_step: int=field(
        default=120,
        metadata={"help":"Save steps"}
    )
    use_gradient_checkpointing: Optional[bool]=field(
        default=False,
        metadata={"help":"Idk man, help me too"}
    )
@dataclass
class PEFTArguments:
    adapter_type: str=field(
        default="LoRA",
        metadata={"help":"The choice of PEFT"}
    )
    _valid_adapter_types: ClassVar[List[str]]=["LoRA","AdaLoRA","Prefix-Tuning"]
    lora_r: Optional[int]=field(
        default=32,
        metadata={"help":"LoRA rank for LoRA"}
    )
    lora_alpha: Optional[int]=field(
        default=64,
        metadata={"help":"LoRA alpha for LoRA"}
    )
    lora_dropout: Optional[float]=field(
        default=0.05
    )
    target_modules: Optional[str]=field(
        default='["q_proj","k_proj","v_proj","up_proj","down_proj"]',
        metadata={"help":"Layers to apply PEFT modules"}
    )
    initial_rank: Optional[int]=field(
        default=32,
        metadata={"help":"Initial rank for AdALoRA"}
    )
    target_rank: Optional[int]=field(
        default=48,
        metadata={"help":"Target rank for AdaLoRA"}
    )
    use_dora: bool=field(
        default=False,
        metadata={"help":"Set it true for DoRA, rn DoRA only supports ConV2d and Linear Layers"}
    )
    use_rslora: bool=field(
        default=False,
        metadata={"help":"Whether to use rank-stabilized LoRA or not"}
    )
    def __post_init__(self):
        if self.adapter_type not in self._valid_adapter_types:
            raise ValueError(
                f"Invalid adapter_type: '{self.adapter_type}'. "
                f"Must be one of {self._valid_adapter_types}."
            )
@dataclass
class LLMArguments:
    train_on_inputs: Optional[bool]=field(
        default=True,
        metadata={"help":"If false, masks out the inputs in loss"}
    )
    group_by_length: Optional[bool]=field(
        default=False,
        metadata={"help":"faster, but produces an odd training curve"}
    )
def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    set_seed(seed)
    os.environ['PYTHONHASHSEED']= '0'
    
def main():
    parser=HfArgumentParser((DataArguments,ModelArguments,TrainingArguments,PEFTArguments,LLMArguments))
    data_args,model_args,training_args,peft_args,llm_args=parser.parse_args_into_dataclasses()
    fix_seed(42)
    gradient_accumulation_steps=training_args.batch_size//training_args.micro_batch_size
    device_map=0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    model=AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch.float32,
        device_map=device_map,
    )
    print(model)
    tokenizer=AutoTokenizer.from_pretrained(model_args.base_model,trust_remote_code=True)
    tokenizer.pad_token_id=(
        0
    )
    target_modules_list = json.loads(peft_args.target_modules)
    tokenizer.padding_side="left"
    if peft_args.adapter_type=="LoRA":
        config=LoraConfig(
            r=peft_args.lora_r,
            lora_alpha=peft_args.lora_alpha,
            target_modules=target_modules_list,
            use_dora=peft_args.use_dora,
            use_rslora=peft_args.use_rslora,
            lora_dropout=peft_args.lora_dropout,
            fan_in_fan_out=False
        )
    model=get_peft_model(model,config)
    data=load_dataset("json",data_files=f"/home/arinjay/model-merging-adapters/dataset/{data_args.task_type}s/{data_args.task_niche}/{data_args.task_name}/train.json")
    model.print_trainable_parameters()
    generate_and_tokenize_prompt_with_tokenizer = lambda data_point: generate_and_tokenize_prompt(
        data_point, tokenizer=tokenizer, cutoff_len=data_args.cutoff_len, add_eos_token=False
    )
    if data_args.val_set_size>0:
        train_val=data["train"].train_test_split(
            test_size=data_args.val_set_size,shuffle=True,seed=training_args.seed
        )
        train_data=(
            train_val["train"].shuffle().map(generate_and_tokenize_prompt_with_tokenizer)
        )
        val_data=(
            train_val["test"].shuffle().map(generate_and_tokenize_prompt_with_tokenizer)
        )
    else:
        train_data=data["train"].shuffle().map(generate_and_tokenize_prompt_with_tokenizer)
        val_data=None
    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=training_args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=training_args.num_epochs,
            learning_rate=training_args.learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if training_args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=training_args.eval_step if training_args.val_set_size > 0 else None,
            save_steps=training_args.save_step,
            output_dir=training_args.output_model,
            save_total_limit=3,
            load_best_model_at_end=True if training_args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=llm_args.group_by_length,
            #report_to="wandb" if use_wandb else None,
            #run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(model_args.output_model)
    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )
fire.Fire(main)