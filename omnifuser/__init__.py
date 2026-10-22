"""OmniFuser research reproduction."""
from __future__ import annotations

import platform
import sys
import threading

__version__ = "0.1.0"

WMI_TIMEOUT_SECONDS = 5.0


def _guard_windows_wmi(timeout: float = WMI_TIMEOUT_SECONDS) -> bool:
    """Keep a stuck Windows WMI service from blocking ``import torch`` forever.

    On Windows, Python 3.12+ asks WMI for the OS version and machine type
    (``platform.uname``), and torch calls it on import. When the WMI service hangs,
    that call never returns and every run stalls at 0 % CPU. The query runs once here
    in a background thread; if it has not answered within ``timeout`` seconds,
    ``platform`` is switched to its own fallback (``sys.getwindowsversion``).
    Returns True when the fallback was enabled.
    """
    if sys.platform != "win32" or not hasattr(platform, "_wmi_query"):
        return False
    probe = threading.Thread(target=platform.uname, name="omnifuser-wmi-probe", daemon=True)
    probe.start()
    probe.join(timeout)
    if not probe.is_alive():
        return False

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("WMI did not respond; using platform's fallback")

    platform._wmi_query = unavailable  # pyright: ignore[reportAttributeAccessIssue]
    print(
        f"omnifuser: the Windows WMI service did not answer within {timeout:.0f} s; "
        "using Python's fallback for platform details. Restarting the service "
        "(Restart-Service winmgmt -Force, as administrator) fixes this.",
        file=sys.stderr,
        flush=True,
    )
    return True


_guard_windows_wmi()
