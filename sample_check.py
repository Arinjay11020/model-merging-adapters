from mergers import *
from transformers import AutoModelForCausalLM
from peft import PeftModel,PeftConfig
import torch
model=AutoModelForCausalLM.from_pretrained("/home/models/Meta-Llama-3.2-3B-Instruct",device_map="cuda:3")
#model_weights=model.state_dict()
#model_code_config=PeftConfig.from_pretrained("/home/arinjay/model-merging-adapters/models/expert/code/lora/llama-3.2-3B-code/final/")
# torch.save(model_weights,"/home/arinjay/model-merging-adapters/models/base_torch/llama3_weights.pth")
# new_model=torch.load("/home/arinjay/model-merging-adapters/models/base_torch/llama3_weights.pth")
model_code=PeftModel.from_pretrained(model,"/home/arinjay/model-merging-adapters/models/expert/code/lora/llama-3.2-3B-code/final/")
final=model_code.merge_and_unload()
TaskV=TaskArithmetic(model,final)
k_model=TaskV.apply_to(model,scaling_coeff=1.0)
flag=True
with torch.no_grad():
    for key in k_model.state_dict():
        if not torch.equal(k_model.state_dict()[key],final.state_dict()[key]):
            flag=False
            break
print(flag)