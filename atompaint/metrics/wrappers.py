from torch.nn import Module, ModuleDict

class TrainValTestMetrics(Module):

    def __init__(self, metrics_factory):
        super().__init__()

        # Use the `_` suffix because all modules have a `train` attribute, so 
        # that can't be used as a module name.
        self.metrics = ModuleDict({
            loop: ModuleDict(metrics_factory())
            for loop in ['train_', 'val_', 'test_']
        })

    def get(self, loop, metric=None):
        metrics = self.metrics[f'{loop}_']

        if metric is None:
            return metrics.items()
        else:
            return metrics[metric]
    
    def log(self, module, loop, *args, **kwargs):
        for name, metric in self.get(loop):
            metric(*args, **kwargs)
            self.log(f'{loop}/{name}', metric)


