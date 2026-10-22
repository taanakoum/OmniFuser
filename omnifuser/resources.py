"""How many CPU threads to use without crowding out other programs.

PyTorch starts one thread per core by default. On a shared laptop or login node that
oversubscribes the CPU: another program's work and ours slow each other down. The
count chosen here is the smallest of

* an explicit ``OMNIFUSER_CPU_THREADS`` (used alone when set),
* the CPUs this process may run on (on Linux, the cores a SLURM job was granted),
* ``OMP_NUM_THREADS`` when set,
* the CPUs currently idle, except inside a SLURM job, whose cores are dedicated,
* a caller's limit (more threads do not speed up this small model).

Idle CPUs are measured without WMI: from the load average on Linux and from
``GetSystemTimes`` sampled over a quarter of a second on Windows.
"""
from __future__ import annotations

import math
import os
import sys
import time


def available_cpus() -> int:
    """CPUs this process may run on."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return max(1, len(affinity(0)))
    return max(1, os.cpu_count() or 1)


def _windows_idle_fraction(seconds: float) -> float | None:
    import ctypes
    from ctypes import wintypes

    def sample() -> tuple[int, int] | None:
        idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
        if not ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None

        def ticks(value: wintypes.FILETIME) -> int:
            return (value.dwHighDateTime << 32) | value.dwLowDateTime

        # Kernel time includes idle time.
        return ticks(idle), ticks(kernel) + ticks(user)

    first = sample()
    time.sleep(seconds)
    second = sample()
    if first is None or second is None or second[1] <= first[1]:
        return None
    return (second[0] - first[0]) / (second[1] - first[1])


def idle_cpus(sample_seconds: float = 0.25) -> float | None:
    """CPUs currently idle across the machine, or None when it cannot be measured."""
    total = os.cpu_count() or 1
    try:
        if sys.platform == "win32":
            fraction = _windows_idle_fraction(sample_seconds)
            return None if fraction is None else max(0.0, min(1.0, fraction)) * total
        load = os.getloadavg()[0]
    except (AttributeError, OSError, ValueError):
        return None
    return max(0.0, total - load)


def cpu_threads(limit: int = 4) -> int:
    """Threads for CPU work now: see the module docstring for the rule."""
    requested = os.environ.get("OMNIFUSER_CPU_THREADS", "")
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    threads = min(limit, available_cpus())
    omp = os.environ.get("OMP_NUM_THREADS", "")
    if omp.isdigit() and int(omp) > 0:
        threads = min(threads, int(omp))
    if "SLURM_JOB_ID" not in os.environ:
        idle = idle_cpus()
        if idle is not None:
            threads = min(threads, math.floor(idle))
    return max(1, threads)
