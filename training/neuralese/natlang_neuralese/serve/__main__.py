"""Serve one complete native shared-state checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import load_native_engine
from .http import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native Neuralese shared-state server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port; the chosen port is printed")
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:N], or auto: cuda when it has room, else cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default=None,
                        help="model weight and cache dtype (default: float32 on CPU, bfloat16 on CUDA)")
    parser.add_argument("--native-checkpoint", required=True,
                        help="complete native shared-state model checkpoint")
    parser.add_argument("--base", default=None, help="exact source model snapshot for native restoration")
    parser.add_argument('--native-participant', help='model identity selected from a complete shared semantic checkpoint')
    parser.add_argument('--native-actor-view', help='optional source-bound compact inference receipt')
    parser.add_argument('--native-actor-view-sha256', help='exact independently bound inference receipt SHA-256')
    parser.add_argument('--semantic-adapter', help='identity-bound shared semantic adapter export receipt')
    parser.add_argument("--prompt-rendering", choices=("full", "program-header"), default=None,
                        help="serving prompt policy (direct artifacts bind this in their manifest; base default: full)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("-c", "--ctx-size", type=int, default=None,
                        help="served context in tokens (default and maximum: the model's max_position_embeddings); "
                             "view plans its write sites against it")
    parser.add_argument("--memory-gb", type=float, default=float(os.environ["NATLANG_CUDA_MEMORY_GB"]) if os.environ.get("NATLANG_CUDA_MEMORY_GB") else None,
                        help="cap this process's CUDA allocations (memory is shared with the rest of the machine)")
    parser.add_argument("--store-dir", nargs="?", const="", default=None, metavar="DIR",
                        help="keep blocks, holds and pins on disk so they outlive restarts (serve/store.py); without DIR, "
                             "<data_nvme>/neuralese-blocks/<dialect> (NVMe)")
    parser.add_argument("--store-resident-gb", type=float, default=2.0,
                        help="with --store-dir: payload memory kept resident (least recently used blocks are evicted)")
    parser.add_argument("--guidance", default=None,
                        help="guidance for requests that do not set one: JSON (serve/guidance.py), e.g. '{\"repeat\": 3}' or true")
    args = parser.parse_args(argv)
    if bool(args.native_actor_view) != bool(args.native_actor_view_sha256) or (args.native_actor_view and not args.native_checkpoint):
        parser.error('native actor view requires its receipt SHA-256 and original full native checkpoint')
    actor_view = None
    if args.native_actor_view:
        from pathlib import Path
        actor_view = {'path':str(Path(args.native_actor_view).resolve(strict=True)),
                      'sha256':args.native_actor_view_sha256,'bytes':Path(args.native_actor_view).stat().st_size}
    import torch

    model_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}.get(args.dtype)

    from natlang_neuralese.devices import auto_device, cap_cuda_memory

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = auto_device()
    cap_cuda_memory(args.device, args.memory_gb)
    engine = load_native_engine(args.native_checkpoint, base=args.base, device=args.device, dtype=model_dtype,
                                prompt_rendering=args.prompt_rendering or "full", participant=args.native_participant,
                                semantic_adapter=args.semantic_adapter, actor_view=actor_view)
    if args.ctx_size:
        engine.context = min(engine.context, args.ctx_size)
    if args.store_dir is not None:
        from ..common.paths import resolve
        from .store import TensorStore

        directory = args.store_dir or resolve("data_nvme", "neuralese-blocks", re.sub(r"[^A-Za-z0-9._-]+", "_", engine.dialect))
        engine.store = TensorStore(directory, int(args.store_resident_gb * 2**30))  # the loaders return an empty store
    engine.default_guidance = json.loads(args.guidance) if args.guidance else None
    engine.start()
    server = serve(engine, args.host, args.port)
    from ..data.prompt_rendering import prompt_rendering_identity
    print(json.dumps({"listening": f"http://{server.server_address[0]}:{server.server_address[1]}",
                      "dialect": engine.dialect, "cutoff": None,
                      "architecture": getattr(engine, "model_config", {}).get("architecture"),
                      "native_checkpoint": getattr(engine, "native_checkpoint", None), "device": args.device,
                      "model_dtype": str(next(engine.backbone.parameters()).dtype),
                      "prompt_rendering": engine.prompt_rendering,
                      "prompt_rendering_identity_sha256":
                          prompt_rendering_identity(engine.prompt_rendering)["identity_sha256"]}), flush=True)
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
