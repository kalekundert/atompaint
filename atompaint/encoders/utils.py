def toggle_pool(f):
    from functools import wraps

    @wraps(f)
    def wrapper(*args, pool, pool_factor, **kwargs):
        return f(*args, pool_factor=pool_factor if pool else 1, **kwargs)

    return wrapper


