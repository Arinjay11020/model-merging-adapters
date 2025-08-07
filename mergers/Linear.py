import torch
class Linear:
    def __init__(self,finetuned,weight_dict=None,scale_coeff=1.0):
        if weight_dict is not None:
            self.weight_dict=weight_dict
        else:
            assert finetuned is not None
            with torch.no_grad():
                self.weight_dict=finetuned.state_dict()
        self.scale_coeff=scale_coeff
    def __add__(self,other):
        with torch.no_grad():
            new_dict={}
            for key in self.weight_dict:
                if key not in other.weight_dict:
                    print(f"Warning, key {key} not present in both models")
                    continue
                new_dict[key]=self.weight_dict[key]*self.scale_coeff+other.weight_dict[key]*other.scale_coeff
        new_scale_coeff=self.scale_coeff+other.scale_coeff
        return Linear(weight_dict=new_dict,scale_coeff=new_scale_coeff)
    def average(self):
        new_dict={}
        with torch.no_grad():
            new_dict={key:val/self.scale_coeff for key, val in self.weight_dict}
        return Linear(weight_dict=new_dict,scale_coeff=1.0)
    def __radd__(self,other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)
    def apply_to(self,base):
        with torch.no_grad():
            base.load_state_dict(self.weight_dict,strict=False)
            return base
            
                