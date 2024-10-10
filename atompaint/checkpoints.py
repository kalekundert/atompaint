import torch
import os

from xxhash import xxh32_hexdigest
from pathlib import Path
from typing import Optional, Literal

class EvalModeCheckpointMixin:
    """
    Put the model into "evaluation mode" before saving or loading the state 
    dict, as recommended for convolutional_ and linear_ layers.

    .. _convolutional: https://quva-lab.github.io/escnn/api/escnn.nn.html#rdconv
    .. _linear: https://quva-lab.github.io/escnn/api/escnn.nn.html#escnn.nn.Linear
    """

    # KBK: I think this functionality should be moved into ESCNN itself, but 
    # for now I'm too lazy to put together a PR.

    def state_dict(self, **kwargs):
        is_training = self.training

        if is_training:
            self.eval()

        state_dict = super().state_dict(**kwargs)

        if is_training:
            self.train()

        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        is_training = self.training

        if is_training:
            self.eval()

        super().load_state_dict(state_dict, **kwargs)

        if is_training:
            self.train()

def load_model_weights(
        model,
        path,
        *,
        prefix,
        device=None,
        xxh32sum: Optional[str] = None,
        mode: Literal['eval', 'train'] = 'eval',
):
    ckpt_path = Path(os.environ['AP_MODEL_WEIGHTS']) / path

    if xxh32sum:
        ckpt_bytes = ckpt_path.read_bytes()
        actual_xxh32sum = xxh32_hexdigest(ckpt_bytes)
        if xxh32sum != actual_xxh32sum:
            raise RuntimeError(f"weights file has the wrong hash\n• path: {ckpt_path}\n• expected xxh32sum: {xxh32sum}\n• actual xxh32sum: {actual_xxh32sum}")

    ckpt = torch.load(ckpt_path, map_location=device)
    weights = extract_state_dict(ckpt['state_dict'], prefix)

    if mode == 'eval':
        model.eval()
        model.requires_grad_(False)

    model.load_state_dict(weights)

def extract_state_dict(state_dict, prefix):
    i = len(prefix)
    return {
            k[i:]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
    }
