# KBK: I think this functionality should be moved into ESCNN itself, but for 
# now I'm too lazy to put together a PR.

class EvalModeCheckpointMixin:
    """
    Put the model into "evaluation mode" before saving or loading the state 
    dict, as recommended for convolutional_ and linear_ layers.

    .. _convolutional: https://quva-lab.github.io/escnn/api/escnn.nn.html#rdconv
    .. _linear: https://quva-lab.github.io/escnn/api/escnn.nn.html#escnn.nn.Linear
    """

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
