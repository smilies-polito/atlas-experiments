import json
import os
import sys
import threading

import psutil

_PROC_CHILDREN = os.path.exists(f"/proc/{os.getpid()}/task/{os.getpid()}/children")


def _children_pids(pid):
    if not _PROC_CHILDREN:
        try:
            return [c.pid for c in psutil.Process(pid).children(recursive=True)]
        except psutil.Error:
            return []
    found, stack = [], [pid]
    while stack:
        p = stack.pop()
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for tid in tids:
            try:
                with open(f"/proc/{p}/task/{tid}/children") as fh:
                    kids = [int(x) for x in fh.read().split()]
            except OSError:
                continue
            found.extend(kids)
            stack.extend(kids)
    return found


def _rss(pid):
    with open(f"/proc/{pid}/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _pss_or_rss(proc):
    try:
        return proc.memory_full_info().pss, True
    except (psutil.AccessDenied, AttributeError):
        return proc.memory_info().rss, False


def main():
    target = int(sys.argv[1])
    interval = float(sys.argv[2])
    me = os.getpid()
    try:
        os.nice(19)
    except OSError:
        pass

    stop = threading.Event()

    def wait_for_stop():
        sys.stdin.readline()
        stop.set()

    threading.Thread(target=wait_for_stop, daemon=True).start()

    state = {
        "peak_main_rss": 0,
        "peak_children_pss": 0,
        "peak_tree": 0,
        "max_children": 0,
        "n_samples": 0,
        "children_pss_exact": True,
    }
    cpu_first, cpu_last, procs = {}, {}, {}

    def sample():
        try:
            main_rss = _rss(target)
        except OSError:
            return
        children = [p for p in _children_pids(target) if p != me]
        children_mem = 0
        for pid in children:
            try:
                proc = procs.setdefault(pid, psutil.Process(pid))
                mem, exact = _pss_or_rss(proc)
                cpu = proc.cpu_times()
            except psutil.Error:
                continue
            children_mem += mem
            state["children_pss_exact"] &= exact
            used = cpu.user + cpu.system
            cpu_first.setdefault(pid, used)
            cpu_last[pid] = used
        state["peak_main_rss"] = max(state["peak_main_rss"], main_rss)
        state["peak_children_pss"] = max(state["peak_children_pss"], children_mem)
        state["peak_tree"] = max(state["peak_tree"], main_rss + children_mem)
        state["max_children"] = max(state["max_children"], len(children))
        state["n_samples"] += 1

    sample()
    print("ready", flush=True)
    while not stop.is_set():
        sample()
        stop.wait(interval)
    sample()

    state["cpu_children"] = sum(cpu_last[p] - cpu_first[p] for p in cpu_last)
    print(json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
