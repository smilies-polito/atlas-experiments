import ctypes
import gc
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import pandas as pd

_MONITOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_monitor.py")
_GB = 1024**3


def _release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _status_bytes(field):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def _reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _cpu_self():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def benchmark_function(function, *args, interval=0.01, **kwargs):
    _release_memory()

    monitor = subprocess.Popen(
        [sys.executable, _MONITOR, str(os.getpid()), str(interval)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    if monitor.stdout.readline().strip() != "ready":
        monitor.kill()
        raise RuntimeError("memory monitor failed to start")

    baseline = _status_bytes("VmRSS")
    hwm_exact = _reset_peak_rss()
    cpu0 = _cpu_self()
    start = time.perf_counter()
    try:
        result = function(*args, **kwargs)
    finally:
        wall = time.perf_counter() - start
        cpu_main = _cpu_self() - cpu0
        hwm = _status_bytes("VmHWM")
        out, _ = monitor.communicate(input="stop\n", timeout=120)

    tree = json.loads(out.strip().splitlines()[-1])
    peak_main = hwm if hwm_exact else tree["peak_main_rss"]
    peak_total = max(peak_main, tree["peak_tree"])

    metrics = {
        "wall_time_s": wall,
        "cpu_time_s": cpu_main + tree["cpu_children"],
        "cpu_main_s": cpu_main,
        "cpu_children_s": tree["cpu_children"],
        "baseline_rss_gb": baseline / _GB,
        "peak_main_gb": peak_main / _GB,
        "peak_children_gb": tree["peak_children_pss"] / _GB,
        "peak_total_gb": peak_total / _GB,
        "peak_delta_gb": (peak_total - baseline) / _GB,
        "peak_main_exact": hwm_exact,
        "children_pss_exact": tree["children_pss_exact"],
        "max_children": tree["max_children"],
        "monitor_samples": tree["n_samples"],
    }
    return result, metrics


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _compact(cpus):
    ranges, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            ranges.append(f"{start}-{c}" if c != start else str(c))
            start = None
    return ",".join(ranges)


def environment_info(packages=("atlas", "palantir", "cellrank", "muon", "anndata", "scanpy",
                               "numpy", "scipy", "scikit-learn", "pandas", "joblib", "petsc4py")):
    info = {
        "node": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpus_available": len(os.sched_getaffinity(0)),
        "total_ram_gb": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / _GB,
        "cpu_affinity": _compact(sorted(os.sched_getaffinity(0))),
        "loadavg_1m_at_start": os.getloadavg()[0],
        "container": os.environ.get("BENCH_CONTAINER", ""),
        "container_sha256": os.environ.get("BENCH_CONTAINER_SHA256", ""),
        "python": platform.python_version(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    for pkg in packages:
        try:
            info[f"v_{pkg}"] = version(pkg)
        except PackageNotFoundError:
            info[f"v_{pkg}"] = ""
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        info["blas"] = ";".join(f"{p.get('internal_api')}:{p.get('num_threads')}" for p in pools)
    except ImportError:
        info["blas"] = ""
    return info


def write_row(row, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    pd.DataFrame([row]).to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, path)
