import torch
import math
import TaskArithmetic
class TIES(TaskArithmetic):
    def __init__(self,base,finetuned,vector=None,top_k=0.5):
        super().__init__(base,finetuned,vector)
        if vector is None:
            assert top_k>0.0 and top_k<=1.0
            self.top_k=top_k
            self.sign={}
            self.trimmed_vector={}
            self.select_top_k()
        else:
            self.trimmed_vector=None
            self.sign=None
    def select_top_k(self):
        count=0
        with torch.no_grad():
            count=sum(t.numel() for t in self.vector.values())
            top_k_amt=int(math.ceil(count*self.top_k))
            all_params=torch.cat([t.abs().flatten() for t in self.vector.values()])
            threshold=torch.topk(all_params, top_k_amt, largest=True).values[-1]
            for key in self.vector:
                param=self.vector[key]
                mask=param.abs()>=threshold
                param=param*mask
                self.trimmed_vector[key]=param
                self.sign[key]=torch.sign(param)
    def elect_final_signs(self,other_signs_list):
        final_signs={}
        all_signs_list=[self.sign]+other_signs_list
        with torch.no_grad():
            for key in self.sign:
                sign_sum=torch.sum(torch.stack([s[key].to(self.sign[key].device) for s in all_signs_list]),dim=0)
                final_signs[key]=torch.sign(sign_sum)
        return final_signs
    def disjoint_merging(self,other_vectors_list, final_signs):
        merged_vector={}
        all_vectors_list=[self.trimmed_vector]+other_vectors_list
        with torch.no_grad():
            for key in self.trimmed_vector:
                sum_agreeing_values=torch.zeros_like(self.trimmed_vector[key])
                agreeing_model_counts=torch.zeros_like(self.trimmed_vector[key],dtype=torch.int)
                for i in range(len(all_vectors_list)):
                    vector=all_vectors_list[i][key]
                    sign_vector=torch.sign(vector)
                    agreement_mask=(sign_vector==final_signs[key]) & (final_signs[key]!=0)
                    sum_agreeing_values[agreement_mask]+=vector[agreement_mask]
                    agreeing_model_counts[agreement_mask]+=1
                merged_vector[key]=torch.where(agreeing_model_counts>0,sum_agreeing_values/agreeing_model_counts,torch.zeros_like(self.trimmed_vector[key]))
        return merged_vector
    def __str__(self):
        super().__str__()
    def __add__(self,other):
        if self.trimmed_vector is None:
            return super().__add__(other)
        final_signs=self.elect_final_signs([other.sign])
        merged_vector=self.disjoint_merging([other.trimmed_vector],final_signs)
        return TIES(vector=merged_vector,top_k=self.top_k)
    def __radd__(self,other):
        super().__radd__(other)
    def __neg__(self):
        if self.trimmed_vector is not None:
            with torch.no_grad():
                new_vector={key: -self.trimmed_vector[key] for key in self.trimmed_vector}
        else:
            with torch.no_grad():
                new_vector={key: -self.vector[key] for key in self.vector}
        return TIES(vector=new_vector,top_k=self.top_k)
    def apply_to(self, base, scaling_coeff=1):
        super().apply_to(base, scaling_coeff)
    
        