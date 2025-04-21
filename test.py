from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import torch.nn as nn
device=torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
