import torch

def get_importance(curr_grads, prev_grads):
    abs_grad_diff = torch.abs(curr_grads - prev_grads).sum()