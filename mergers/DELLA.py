from TaskArithmetic import *
class DELLA(TaskArithmetic):
    def __init__(self,base,finetuned,vector=None,p=None,epsilon=None):
        super().__init__(base,finetuned,vector)
    def __add__(self,other):
        pass
    def __radd__(self,other):
        pass
    def __neg__(self):
        pass
    def __str__(self):
        pass
    def apply_to(self, base, scaling_coeff=1):
        return super().apply_to(base, scaling_coeff)