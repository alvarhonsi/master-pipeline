from contextlib import contextmanager
import torch

@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor
        
    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)