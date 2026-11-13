import torch
import torch.nn as nn

# Bug A: optimizer.zero_grad() called only inside the train branch of run_epoch
# but self.optimizer is created in __init__. In train() mode we zero grads before
# backward -- OK. But check: does PrimaryTrainer freeze for error mapper?
print("Bug A check: zero_grad exists on optimizer:", hasattr(torch.optim.Adam([nn.Parameter(torch.tensor([1.0]) RequiresGrad=True)]), 'zero_grad') is False)

# Actually test real optimizer
opt = torch.optim.Adam([nn.Parameter(torch.tensor([1.0], requires_grad=True))])
print("zero_grad callable:", callable(getattr(opt, 'zero_grad', None)))
