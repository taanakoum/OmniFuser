"""Keep the test suite inside the CPUs that are actually free.

PyTorch starts one thread per core by default. On a shared or busy laptop that
oversubscribes the CPU and makes the small test models slower, not faster. The tests
use the rule in ``omnifuser.resources`` (at most 4 threads, never more than the idle
CPUs, ``OMP_NUM_THREADS`` or a SLURM allocation). Set ``OMNIFUSER_TEST_THREADS`` to
choose another count, e.g. 1 while another job holds every core.
"""
from __future__ import annotations

import os

import omnifuser  # noqa: F401  # first: guards against a stuck Windows WMI service

# isort: split
import torch

from omnifuser.resources import cpu_threads

_requested = os.environ.get("OMNIFUSER_TEST_THREADS", "")
torch.set_num_threads(int(_requested) if _requested.isdigit() and int(_requested) > 0 else cpu_threads(limit=4))
