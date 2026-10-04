"""`python -m natlang_neuralese.serve --port 8090 [--device cuda] [--heads checkpoint.pt]`"""

from __future__ import annotations

import argparse
import json
import sys

from . import load_engine
from .http import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neuralese reference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port; the chosen port is printed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--base", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--heads", default=None, help="S3 trainer checkpoint with port heads and control rows")
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--max-block", type=int, default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-gb", type=float, default=None,
                        help="cap this process's CUDA allocations (memory is shared with the rest of the machine)")
    args = parser.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    if args.memory_gb and args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    engine = load_engine(args.base, args.lora, args.heads, args.cutoff, args.max_block, args.device, args.dialect)
    engine.start()
    server = serve(engine, args.host, args.port)
    print(json.dumps({"listening": f"http://{server.server_address[0]}:{server.server_address[1]}",
                      "dialect": engine.dialect, "cutoff": engine.heads.cutoff}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
