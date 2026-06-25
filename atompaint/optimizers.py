import torch.nn as nn


def make_param_groups(model: nn.Module, weight_decay: float = 1e-2) -> list:
    """
    Split model parameters into two AdamW param groups: one with weight decay
    (conv/linear weights) and one without (batch norm parameters and biases).

    The split is based on parameter names: `.weight` → decay, `.bn` or
    anything else → no decay.
    """
    DECAY, NO_DECAY = 0, 1
    param_groups = [
        {'params': [], 'weight_decay': weight_decay},
        {'params': [], 'weight_decay': 0.0},
    ]

    for name, param in model.named_parameters():
        if '.bn' in name:
            i = NO_DECAY
        elif '.weight' in name:
            i = DECAY
        else:
            i = NO_DECAY
        param_groups[i]['params'].append(param)

    return param_groups
