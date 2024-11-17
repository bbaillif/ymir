import torch
from torch.distributions import Categorical

class CategoricalMasked(Categorical):
    
    def __init__(self, 
                 probs: torch.Tensor = None, 
                 logits: torch.Tensor = None, 
                 validate_args: bool = None, 
                 masks: torch.Tensor = None):
        assert (probs is not None) or (logits is not None)
        self.masks = masks
        if self.masks is not None:
            if probs is not None:
                assert masks.size() == probs.size()
            elif logits is not None:
                try:
                    assert masks.size() == logits.size()
                except:
                    import pdb;pdb.set_trace()
            self.mask_value = torch.tensor(torch.finfo(logits.dtype).min, 
                                           dtype=logits.dtype)
            logits = torch.where(self.masks, logits, self.mask_value)
            
        super().__init__(probs, 
                        logits, 
                        validate_args)
    
    def entropy(self):
        if self.masks is None:
            return super().entropy()
        min_real = torch.finfo(self.logits.dtype).min
        logits = torch.clamp(self.logits, min=min_real)
        p_log_p = logits * self.probs
        zero_value = torch.tensor(0, dtype=p_log_p.dtype, 
                                           device=p_log_p.device)
        p_log_p = torch.where(self.masks, 
                              p_log_p, 
                              zero_value)
        return -p_log_p.sum(-1)