import torch
class SLERP:
    def __init__(self,finetuned_1,finetuned_2,eps=1e-6,t=0.5):
        assert finetuned_1 is not None and finetuned_2 is not None
        with torch.no_grad():
            self.weight_dict_1=finetuned_1.state_dict()
            self.weight_dict_2=finetuned_2.state_dict()
        self.eps=eps
        self.t=t
        self.merged_dict={}
        self.process()
    def flatten_state_dict(self,state_dict):
        shapes,dtypes,flattened_tensor={},{},[]
        with torch.no_grad():
            for key in state_dict:
                shapes[key]=state_dict[key].shape
                dtypes[key]=state_dict[key].dtype
                flattened_tensor.append(state_dict[key].view(-1))
        return torch.cat(flattened_tensor), shapes, dtypes
    def unflatten_tensor(self,vector,shapes,dtypes):
        state_dict,current_idx={},0
        with torch.no_grad():
            for key in shapes:
                num_elements=shapes[key].numel()
                state_dict[key]=vector[current_idx:current_idx+num_elements].view(shapes[key]).to(dtypes[key])
                current_idx+=num_elements
        return state_dict
    def slurp(self,v0,v1):
        v0_normalised=v0/(v0.norm()+self.eps)
        v1_normalised=v1/(v1.norm()+self.eps)
        dot=torch.sum(v0_normalised*v1_normalised)
        theta_0=torch.acos(dot)
        sin_theta_0=torch.sin(theta_0)
        if sin_theta_0.abs()<self.eps:
            return (1-self.t)*v0+self.t*v1
        theta_t=theta_0*self.t
        sin_theta_t=torch.sin(theta_t)
        s0=torch.sin(theta_0-theta_t)/sin_theta_0
        s1=sin_theta_t/sin_theta_0
        res=s0*v0+s1*v1
        return res
    def process(self):
        with torch.no_grad():
            v0,shapes,dtypes=self.flatten_state_dict(self.weight_dict_1)
            v1,shapes_1,dtypes_1=self.flatten_state_dict(self.weight_dict_2)
            interpolated_vec=self.slurp(v0,v1)
            self.merged_dict=self.unflatten_tensor(interpolated_vec,shapes,dtypes)
    def apply_to(self,base):
        with torch.no_grad():
            base.load_state_dict(self.merged_dict,strict=False)
            return base
            