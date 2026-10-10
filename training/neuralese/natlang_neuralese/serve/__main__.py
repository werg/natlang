"""`python -m natlang_neuralese.serve --port 8090 [--device cuda] [--heads checkpoint.pt] [--store-dir [DIR]]`"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import load_engine
from .http import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Neuralese reference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port; the chosen port is printed")
    parser.add_argument("--device", default="cpu", help="cpu, cuda[:N], or auto: cuda when it has room, else cpu")
    parser.add_argument("--base", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--heads", default=None, help="S3 trainer checkpoint with port heads and control rows")
    parser.add_argument("--recurrence-checkpoint", default=None,
                        help="complete learned recurrent checkpoint; requires its exact runtime qualification")
    parser.add_argument("--runtime-qualification", default=None,
                        help="passed recurrence_runtime report binding the exact checkpoint")
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--max-block", type=int, default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--deterministic-gradients", action="store_true",
                        help="gradient replay under torch.use_deterministic_algorithms (same request, same gradient bytes on a GPU; "
                             "slower, and an op without a deterministic kernel fails); same as NATLANG_NEURALESE_DETERMINISTIC=1")
    parser.add_argument("-c", "--ctx-size", type=int, default=None,
                        help="served context in tokens (default and maximum: the model's max_position_embeddings), as "
                             "the fork's -c; view plans its write sites against it")
    parser.add_argument("--prefill-padding", action="store_true",
                        help="pack prompts of different lengths into one left-padded prefill (faster, not bit-identical)")
    parser.add_argument("--memory-gb", type=float, default=float(os.environ["NATLANG_CUDA_MEMORY_GB"]) if os.environ.get("NATLANG_CUDA_MEMORY_GB") else None,
                        help="cap this process's CUDA allocations (memory is shared with the rest of the machine)")
    parser.add_argument("--projection", action="append", default=[], metavar="NAME=PATH",
                        help="an adapter projection P (model/projections.py) served under NAME; repeatable")
    parser.add_argument("--store-dir", nargs="?", const="", default=None, metavar="DIR",
                        help="keep blocks, holds and pins on disk so they outlive restarts (serve/store.py); without DIR, "
                             "<data_nvme>/neuralese-blocks/<dialect> (NVMe)")
    parser.add_argument("--store-resident-gb", type=float, default=2.0,
                        help="with --store-dir: payload memory kept resident (least recently used blocks are evicted)")
    parser.add_argument("--guidance", default=None,
                        help="guidance for requests that do not set one: JSON (serve/guidance.py), e.g. '{\"repeat\": 3}' or true")
    args = parser.parse_args(argv)
    if args.deterministic_gradients:
        os.environ["NATLANG_NEURALESE_DETERMINISTIC"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch

    from natlang_neuralese.devices import auto_device, cap_cuda_memory

    torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = auto_device()
    cap_cuda_memory(args.device, args.memory_gb)
    if args.recurrence_checkpoint:
        if not args.runtime_qualification or any(value is not None for value in
                (args.base, args.lora, args.heads, args.cutoff, args.max_block, args.dialect)):
            parser.error('recurrence checkpoint requires runtime qualification and no backbone/port overrides')
        from pathlib import Path
        from ..train.output_embedding_projection import sha
        proof = json.loads(Path(args.runtime_qualification).read_text())
        if (proof.get('schema') != 'natlang.recurrence-runtime-requalification/1'
                or proof.get('runtime_transport_passed') is not True
                or proof.get('input_gradient_equal_direct_raw') is not True
                or proof.get('weights_unmodified') is not True
                or proof.get('device_type') != torch.device(args.device).type
                or proof.get('model_dtype') != str(torch.float32 if args.device == 'cpu' else torch.bfloat16)
                or proof.get('pins', {}).get(str(Path(args.recurrence_checkpoint))) != sha(args.recurrence_checkpoint)):
            parser.error('runtime qualification does not qualify this exact recurrence checkpoint')
        from .recurrence_checkpoint import load_recurrence_checkpoint
        engine, _ = load_recurrence_checkpoint(args.recurrence_checkpoint, device=args.device)
        if proof.get('conv_kernel_enabled') != (getattr(engine.backbone, 'conv_kernel', None) is not None):
            parser.error('convolution execution differs from the qualified runtime')
        engine.recurrence_checkpoint['runtime_requalified'] = True
        engine.recurrence_checkpoint['runtime_report_sha256'] = sha(args.runtime_qualification)
    else:
        if args.runtime_qualification:
            parser.error('runtime qualification requires a recurrence checkpoint')
        engine = load_engine(args.base, args.lora, args.heads, args.cutoff, args.max_block, args.device, args.dialect)
    engine.prefill_padding = args.prefill_padding
    if args.ctx_size:
        engine.context = min(engine.context, args.ctx_size)
    if args.store_dir is not None:
        from ..common.paths import resolve
        from .store import TensorStore

        directory = args.store_dir or resolve("data_nvme", "neuralese-blocks", re.sub(r"[^A-Za-z0-9._-]+", "_", engine.dialect))
        engine.store = TensorStore(directory, int(args.store_resident_gb * 2**30))  # the loaders return an empty store
    engine.default_guidance = json.loads(args.guidance) if args.guidance else None
    if args.projection:
        from ..model.projections import AdapterProjection

        for item in args.projection:
            name, path = item.split("=", 1)
            engine.projections[name] = AdapterProjection.load(path, map_location=args.device).to(args.device).eval()
    engine.start()
    server = serve(engine, args.host, args.port)
    print(json.dumps({"listening": f"http://{server.server_address[0]}:{server.server_address[1]}",
                      "dialect": engine.dialect, "cutoff": engine.heads.cutoff, "device": args.device}), flush=True)
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
