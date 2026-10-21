"""Keep the test suite inside a share of the machine's CPUs.

PyTorch starts one thread per core by default. On a shared or busy laptop that
oversubscribes the CPU and makes the small test models slower, not faster. The tests
use half the logical CPUs (at most 4); set ``OMNIFUSER_TEST_THREADS`` to choose another
count, e.g. 1 while another job holds every core. ``train_fold`` never raises the count.
"""
from __future__ import annotations

import os

import torch

_requested = int(os.environ.get("OMNIFUSER_TEST_THREADS", "0"))
torch.set_num_threads(_requested if _requested > 0 else max(1, min(4, (os.cpu_count() or 2) // 2)))
