"""Streamlit helpers: stream subprocess stdout into session log + progress bar."""

from __future__ import annotations

import queue
import subprocess
import threading
from typing import Callable


def _stream_process_reader(proc: subprocess.Popen, q: queue.Queue) -> None:
    for line in iter(proc.stdout.readline, ""):
        q.put(("line", line.rstrip()))
    proc.wait()
    q.put(("exit", proc.returncode))


def start_stream_reader(proc: subprocess.Popen, q: queue.Queue) -> threading.Thread:
    t = threading.Thread(target=_stream_process_reader, args=(proc, q), daemon=True)
    t.start()
    return t


def drain_subprocess_queue(
    q: queue.Queue,
    *,
    log_lines: list[str],
    render_log: Callable[[str], None],
    progress_bar,
    step_map: dict[str, tuple[int, str]],
    progress_holder: list[int],
    queue_timeout: float = 0.3,
) -> int:
    """
    Process messages until exit. Updates log_lines, calls render_log on each line,
    and advances progress_bar when a step_map key is a substring of a line.
    progress_holder must be a one-element list [current_pct] for monotonic updates.
    Returns process return code.
    """
    while True:
        try:
            msg_type, payload = q.get(timeout=queue_timeout)
        except queue.Empty:
            render_log("\n".join(log_lines[-60:]))
            continue

        if msg_type == "line":
            log_lines.append(payload)
            render_log("\n".join(log_lines[-60:]))
            current = progress_holder[0]
            for keyword, (pct, label) in step_map.items():
                if keyword in payload and pct > current:
                    progress_holder[0] = pct
                    progress_bar.progress(pct, text=label)
                    break
        elif msg_type == "exit":
            return int(payload)
