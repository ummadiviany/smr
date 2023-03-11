import torch

def get_importance(curr_grads, prev_grads):
    abs_grad_diff = 0
    for (n1, p1), (n2, p2) in zip(curr_grads.items(), prev_grads.items()):
        abs_grad_diff += torch.sum(torch.abs(p1 - p2)).item()
        # print(n1, n2)
        # print(p1.shape, p2.shape)
    return abs_grad_diff