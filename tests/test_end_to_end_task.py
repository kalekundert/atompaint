import atompaint.end_to_end.task as ap
import torch
import torch.nn as nn
import lightning as L
import numpy as np

from atompaint.diffusion.karras2022 import InpaintParams
from atompaint.end_to_end.data import collate_end_to_end_samples
from atompaint.classifiers.amino_acid import get_amino_acid_labels
from torch.utils.data import DataLoader
from pytest import approx

def test_sequence_recovery():
    # The goals of this test are to make sure that (i) the code runs without 
    # error and (ii) the actual sequence recovery metric is calculated 
    # correctly.  To do this, we're using a fake classifier that always 
    # predicts the first class.

    # Note that this test doesn't make any effort to make sure that the 
    # inpainting and classification is performed correctly.

    class FakeTask(L.LightningModule):

        def __init__(self):
            super().__init__()
            self.denoiser = FakeDenoiser()
            self.classifier = FakeClassifier()

        def validation_step(self, batch):
            pass

    class FakeDenoiser(nn.Module):

        def __init__(self):
            super().__init__()
            self.x_shape = (6, 11, 11, 11)

        def forward(self, x_noisy, sigma, x_self_cond=None, label=None):
            return torch.zeros_like(x_noisy)

    class FakeClassifier(nn.Module):

        def __init__(self):
            super().__init__()
            self.amino_acids = get_amino_acid_labels()

        def forward(self, x):
            logits = torch.zeros((x.shape[0], 20), device=x.device)
            logits[:, 0] = 1
            return logits

    def fake_sample(labels):
        b = len(labels)
        return dict(
                x_clean=np.zeros((6, 11, 11, 11)),
                x_noise=np.zeros((6, 11, 11, 11)),
                seq_recovery_mask=np.zeros((6, 11, 11, 11)),
                sigma=0,
                aa_crops=b * [
                    (slice(None), slice(None), slice(None), slice(None)),
                ],
                aa_channels=np.zeros((b, 1, 11, 11, 11)),
                aa_labels=np.array(labels),
                use_x_pred=True,
        )

    task = FakeTask()
    task.train()

    # Deliberately make more samples than the callback will use, to make sure 
    # it stops when it should.
    samples = [
            fake_sample([0, 1]),
            fake_sample([2, 3]),
            fake_sample([4, 5]),
            fake_sample([6, 7]),
    ]
    dataloader = DataLoader(
            samples,
            batch_size=2,
            collate_fn=collate_end_to_end_samples,
    )

    trainer = L.Trainer(
            callbacks=[
                ap.SequenceRecovery(
                    dataloader=dataloader,
                    limit_samples=2,
                    inpaint_params=InpaintParams(
                            noise_steps=2,
                            resample_steps=1,
                    ),
                ),
            ],
            logger=False,
            enable_checkpointing=False,
    )
    trainer.validate(
            model=task,
            dataloaders=dataloader,
            ckpt_path=None,
            verbose=False,
    )

    assert trainer.callback_metrics['val/sequence_recovery'] == approx(1/4)

