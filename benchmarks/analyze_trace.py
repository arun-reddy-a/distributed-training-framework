#!/usr/bin/env python3
"""Post-process a PyTorch Profiler Chrome trace into a comm/compute overlap report.

.. code-block:: bash

    python benchmarks/analyze_trace.py traces/trace_ddp_rank0.json
    python benchmarks/analyze_trace.py traces/*.json --csv overlap.csv

Traces come from ``--profile`` on ``examples/train_gpt.py`` or ``--profile-dir``
on ``bench_parallelism.py``, and can equally be opened in ``chrome://tracing``
or ``ui.perfetto.dev`` to look at the same intervals by eye.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from minidist.profiling import analyze_trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traces", nargs="+", type=Path)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--cpu", action="store_true",
                   help="force the CPU-operator view even if kernels are present")
    args = p.parse_args()

    rows = []
    for path in args.traces:
        if not path.exists():
            print(f"  {path}: not found", file=sys.stderr)
            continue
        report = analyze_trace(path, prefer_device=not args.cpu)
        print(f"\n{path.name}")
        print(report.format())
        rows.append((path.name, report))

    if args.csv and rows:
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "trace", "device", "wall_s", "compute_s", "comm_s", "overlapped_s",
                "exposed_comm_s", "overlap_fraction", "exposed_fraction_of_step",
            ])
            for name, r in rows:
                writer.writerow([
                    name, r.device, f"{r.wall_s:.6f}", f"{r.compute_s:.6f}",
                    f"{r.comm_s:.6f}", f"{r.overlapped_s:.6f}", f"{r.exposed_comm_s:.6f}",
                    f"{r.overlap_fraction:.4f}", f"{r.exposed_fraction_of_step:.4f}",
                ])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
