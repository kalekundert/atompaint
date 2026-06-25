import torch
import os

from atompaint.utils import identity
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from xxhash import xxh32_hexdigest
from pathlib import Path
from typing import Optional, Literal, Callable


class WarmupAwareModelCheckpoint(ModelCheckpoint):
    """
    A `ModelCheckpoint` that suppresses top-k checkpoint selection during an
    initial warmup period.

    When training with a regularization warmup schedule (e.g. a KL warmup in a
    VAE), the lowest validation loss is often reached *during* the warmup,
    while the regularization weight is still near zero.  Such a checkpoint
    minimizes a different (and easier) objective than the fully-weighted loss,
    so it is not representative of the final model.  This subclass skips top-k
    saving until `current_epoch >= warmup_epochs`.
    """

    def __init__(self, warmup_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self._warmup_epochs = warmup_epochs

    def _save_topk_checkpoint(self, trainer, monitor_candidates):
        if trainer.current_epoch < self._warmup_epochs:
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)


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

def get_artifact_dir():
    return Path(os.environ['AP_MODEL_WEIGHTS'])

def has_artifact_dir():
    return 'AP_MODEL_WEIGHTS' in os.environ

def load_model_weights(
        model,
        path,
        *,
        fix_keys: Callable[[str], str] = identity,
        xxh32sum: Optional[str] = None,
        mode: Literal['eval', 'train'] = 'eval',
):
    ckpt_path = get_artifact_dir() / path

    if xxh32sum:
        ckpt_bytes = ckpt_path.read_bytes()
        actual_xxh32sum = xxh32_hexdigest(ckpt_bytes)
        if xxh32sum != actual_xxh32sum:
            raise RuntimeError(f"weights file has the wrong hash\n• path: {ckpt_path}\n• expected xxh32sum: {xxh32sum}\n• actual xxh32sum: {actual_xxh32sum}")

    # It doesn't matter what device the tensors returned by `torch.load()` are 
    # on.  `Module.load_state_dict()` copies the values of the these tensors 
    # into the corresponding tensors in the model, but without changing any 
    # other aspect of the model tensors, so the model tensors will stay on 
    # whatever device they started on.
    #
    # Since the device here doesn't matter, it makes sense to map everything to 
    # the CPU.  By default, the tensors in the checkpoint will probably be 
    # loaded onto the GPU, since I do my training on GPUs.  But I'd like the 
    # code to run correctly even on systems without a GPU.  The CPU device is 
    # always available, so it's a better target.

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    weights = extract_state_dict(ckpt['state_dict'], fix_keys=fix_keys)

    if mode == 'eval':
        model.eval()
        model.requires_grad_(False)

    model.load_state_dict(weights)

def extract_state_dict(
        state_dict: dict[str, Tensor],
        *,
        fix_keys: Callable[[str], str] = identity,
):
    out = {}

    for k, v in state_dict.items():
        try:
            out[fix_keys(k)] = v
        except DropKey:
            continue

    return out

def strip_prefix(k, *, prefix):
    if k.startswith(prefix):
        return k[len(prefix):]
    raise DropKey

class DropKey(Exception):
    pass

