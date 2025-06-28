#
# Helper function for the trainer class so its easier to write trainers
#

import torch

def shift_logits_labels(logits, labels):

    if isinstance(logits, torch.Tensor):
        logits = (logits,)
    else:
        logits = tuple(logits)

    flat_logits = [l[..., :-1, :].reshape(-1, l.size(-1)) for l in logits ]

    flat_labels = labels[..., 1:].reshape(-1)

    return (*flat_logits, flat_labels)
