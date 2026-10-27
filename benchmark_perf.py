"""Backward-compatible entry point for the RTX 4090 report/guide.

The benchmark implementation now lives in perf_bench.py; this shim keeps the
`benchmark_perf.py` commands documented in the 4090 test report working.
"""

from perf_bench import main

if __name__ == "__main__":
    main()
