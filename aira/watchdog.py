"""Close this page when the launcher that opened it goes away.

The launcher passes its process id in AIRA_PARENT_PID. A background thread checks
once a second; if the launcher is gone (stopped, closed, or crashed), this page exits.
Pages started from a terminal (no AIRA_PARENT_PID) are not affected.
"""
import os
import sys
import threading
import time


def _alive(pid):
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE, WAIT_TIMEOUT = 0x00100000, 0x00000102
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
        if not h:
            return False
        try:
            return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == WAIT_TIMEOUT
        finally:
            ctypes.windll.kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)  # POSIX: signal 0 only checks existence
        return True
    except OSError:
        return False


def follow_launcher(on_exit=None):
    pid = os.environ.get("AIRA_PARENT_PID")
    if not pid:
        return

    def watch():
        while True:
            time.sleep(1.0)
            if not _alive(int(pid)):
                if on_exit:
                    try:
                        on_exit()
                    except Exception:
                        pass
                os._exit(0)

    threading.Thread(target=watch, daemon=True).start()
