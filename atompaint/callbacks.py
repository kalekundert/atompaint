from lightning import Callback
from fastdigest import TDigest

class LogWeightQuantiles(Callback):

    def __init__(self, max_log_q=4):
        self.max_log_q = max_log_q

    def on_train_epoch_end(self, trainer, pl_module):
        digest = TDigest()

        for param in pl_module.parameters():
            if not param.requires_grad:
                continue

            digest.batch_update(param.flatten().numpy(force=True))

        for q in range(1, self.max_log_q + 1):
            q_lo = 10**-q
            q_hi = 1 - q_lo
            pl_module.log(f'weights/pq={-q}', digest.quantile(q_lo))
            pl_module.log(f'weights/pq={q}', digest.quantile(q_hi))

