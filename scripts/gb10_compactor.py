#!/usr/bin/env python3
"""Keep large free blocks available on the GB10 so CUDA contexts can be created (2026-11-12).

A new CUDA context's buffers come from physically contiguous host pages; when the Normal zone has almost no free
order >= 8 blocks, context creation fails (`cuDevicePrimaryCtxRetain` CUDA_ERROR_OUT_OF_MEMORY, kernel NVRM
NV_ERR_NO_MEMORY) with tens of GB free. Without root this service compacts memory: whenever less than --min-mb is free in
such blocks it faults --gb of THP-advised memory (direct compaction) and releases it
(natlang_neuralese.common.cuda_context.compact). Root fix: vm.compaction_proactiveness.

    gb10_compactor.py [--min-mb 512] [--gb 2] [--interval 15]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/neuralese'))
from natlang_neuralese.common.cuda_context import compact, high_order_free_mb  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--min-mb', type=int, default=512)
    parser.add_argument('--gb', type=float, default=2.0)
    parser.add_argument('--interval', type=float, default=15.0)
    args = parser.parse_args()
    while True:
        free = high_order_free_mb()
        if free is not None and free < args.min_mb:
            print(json.dumps({'time': round(time.time()), 'event': 'compact', **compact(args.gb)}), flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
