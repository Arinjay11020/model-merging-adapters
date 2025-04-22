#Right now trying to add only a few finetuning ways, might be tailored to LLM only
import os
import sys
import fire
import random
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, HFArgumentParser, set_seed
from datasets import load_dataset
from typing import List, Optional, Union, ClassVar
from peft.src.peft import (
    get_peft_config,get_peft_model,
    TaskType,
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification
)
from peft.src.peft.tuners.tuners_utils import BaseTunerLayer
from peft.src.peft.tuners.lora import LoraConfig,LoftQConfig,LoraRuntimeConfig,LoraModel,LoraLayer
from peft.src.peft.tuners.adalora import AdaLoraConfig
from peft.src.peft.tuners.prefix_tuning import PrefixTuningConfig
from dataclasses import dataclass,field
from preprocess_dataset import preprocess_dataset,tokenize,generate_and_tokenize_prompt,generate_prompt
@dataclass
class DataArguments:
    task_name:Optional[list]=field(
        default=None,
        metadata={"help":"The name of the dataset"}
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
        default="models/LLMs/model_name",
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
    _valid_adapter_types: ClassVar[List[str]]=["LoRA","AdaLoRA","DoRA","Prefix-Tuning"]
    lora_r: Optional[int]=field(
        default=32,
        metadata={"help":"LoRA rank for LoRA"}
    )
    lora_alpha: Optional[int]=field(
        default=64,
        metadata={"help":"LoRA alpha for LoRA"}
    )
    lora_droput: Optional[int]=field(
        default=0.05
    )
    target_modules: Optional[Union[list[str], str]]=field(
        default=["q_proj","k_proj","v_proj","up_proj","down_proj"],
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
    use_dora: Optional[bool]=field(
        default=False,
        metadata={"help":"Set it true for DoRA, rn DoRA only supports ConV2d and Linear Layers"}
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
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmarks=False
    set_seed(seed)
    os.environ['PYTHONHASHSEED']= '0'
    
def main():
    parser=HFArgumentParser((DataArguments,ModelArguments,TrainingArguments,PEFTArguments,LLMArguments))
    data_args,model_args,training_args,peft_args,llm_args=parser.parse_args_into_dataclasses()
    set_seed(42)
    gradient_accumulation_steps=training_args.batch_size//training_args.micro_batch_size
    
        
