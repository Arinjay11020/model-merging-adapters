#Right now trying to add only a few finetuning ways, might be tailored to LLM only
import os
import sys
import fire 
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModel,AutoConfig,HFArgumentParser
from datasets import load_dataset
from typing import List, Optional, Union
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
class DataArgs:
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
    pass
