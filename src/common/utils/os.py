# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch


def get_cpu_worker_count() -> int:
    """
    Calculate the number of workers based on the available CPU cores.

    If the CPU count is undetermined, defaults to 1 worker. If there are 2 or fewer CPUs,
    uses 1 worker to avoid system overload. Otherwise, reserves 2 CPUs for system stability
    and uses the rest.

    Returns:
        int: The number of workers to use.
    """
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    else:
        return 1 if cpu_count <= 2 else cpu_count - 2


def set_global_seed(random_state: int) -> None:
    """
    Set the seed for all random number generators.

    Args:
        random_state (int): Seed for random number generator.
    """
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)
