import torch
class TaskArithmetic:
    def __init__(self,base,finetuned,vector=None):
        if vector is not None:
            self.vector=vector
        else:
            assert base is not None and finetuned is not None
            with torch.no_grad():
                base_dict=base.state_dict()
                finetune_dict=finetuned.state_dict()
                self.vector={}
                for key in base_dict:
                    if base_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key]=finetune_dict[key]-base_dict[key]
    def __add__(self,other):
        with torch.no_grad():
            new_vector={}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} not present in both task vectors")
                    continue
                new_vector[key]=self.vector[key]+other.vector[key]
        return TaskArithmetic(vector=new_vector)
    def __radd__(self,other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)
    def __neg__(self):
        with torch.no_grad():
            new_vector={}
            for key in self.vector:
                new_vector[key]=-self.vector[key]
        return TaskArithmetic(vector=new_vector)
    def __str__(self):
        return f"Task Vector for this task: \n {self.vector}",
    def apply_to(self,base,scaling_coeff=1.0):
        with torch.no_grad():
            base_model=base
            new_state_dict={}
            base_dict=base_model.state_dict()
            for key in base_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key]=base_dict[key]+scaling_coeff*self.vector[key]
            base_model.load_state_dict(new_state_dict,strict=False)
            return base_model