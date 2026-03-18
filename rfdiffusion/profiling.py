from contextlib import contextmanager

import torch


@contextmanager
def nvtx_range(name: str, enabled: bool = True):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield
