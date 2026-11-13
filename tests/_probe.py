import torch
x = torch.ones(2,2, requires_grad=True)
with torch.enable_grad():
    y = x.sum()
    print('requires_grad:', y.requires_grad)
print('done')
