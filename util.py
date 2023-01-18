import builtins as __builtin__
import configs

def print(*args, **kwargs):
    if configs.DEBUG:
         return __builtin__.print(*args, **kwargs)
    return
