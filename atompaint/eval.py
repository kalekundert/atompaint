def with_max_batch_size(model, max_batch_size):
    import torch
    import functools
    from more_itertools import sliced

    # For now, we require that the model only has a single input.  This could 
    # be relaxed in the future.

    @functools.wraps(model)
    def wrapped_model(x):
        # If we knew the output dimensions, it'd be a little more efficient to 
        # allocate the output tensor up front and fill it in as we go.  But we 
        # don't, so we just collect the results in a list and concatenate them 
        # at the end.
        y = []

        for x_i in sliced(x, max_batch_size):
            y_i = model(x_i)
            y.append(y_i)

        return torch.cat(y)

    return wrapped_model







