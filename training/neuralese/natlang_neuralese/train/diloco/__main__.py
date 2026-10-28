"""Serve a DiLoCo coordinator root, or print its status.

    python -m natlang_neuralese.train.diloco serve --root R [--host 0.0.0.0] [--port 8080]
    python -m natlang_neuralese.train.diloco status --root R
    python -m natlang_neuralese.train.diloco upload-file --url URL --file PATH --name NAME
    python -m natlang_neuralese.train.diloco download-file --url URL --name NAME --out PATH

upload-file/download-file move large files (learner full states, continuation checkpoints) between sandboxes through
the coordinator's file store (ROOT/uploads/NAME, sha256-verified, resumable, streamed in bounded memory), using the
learner's bearer token and gateway share token ($NATLANG_CW_SECRETS/diloco-token and diloco-gateway).

The bearer token comes from NATLANG_DILOCO_TOKEN or the delivered
$NATLANG_CW_SECRETS/diloco-token file. Trainer-specific seeding and
export live with the trainer (``natlang_neuralese.train.native_diloco``).
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from .coordinator import TOKEN_ENV, TOKEN_SECRET, Coordinator, credential, serve_coordinator
from .worker import GATEWAY_SECRET, GATEWAY_TOKEN_ENV, Transport


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    for name in ("upload-file", "download-file"):
        command = commands.add_parser(name)
        command.add_argument("--url", required=True)
        command.add_argument("--name", required=True)
        command.add_argument("--timeout", type=float, default=3600.0)
        command.add_argument("--allow-http", action="store_true", help="plain HTTP to a private host (tests)")
        if name == "upload-file":
            command.add_argument("--file", type=Path, required=True)
            command.add_argument("--chunk-mib", type=int, default=1024)
        else:
            command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("upload-file", "download-file"):
        transport = Transport(args.url, credential(TOKEN_ENV, TOKEN_SECRET), args.timeout, allow_http=args.allow_http,
                              gateway=credential(GATEWAY_TOKEN_ENV, GATEWAY_SECRET))
        if args.command == "upload-file":
            result = transport.upload_file(args.file, args.name, chunk_bytes=args.chunk_mib << 20)
        else:
            result = transport.download_file(args.name, args.out)
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "status":
        print(json.dumps(Coordinator(args.root).status(), indent=2, sort_keys=True))
        return
    stop = threading.Event()
    # SIGTERM stops serving and merges what learners sent last (Coordinator.close).
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    serve_coordinator(args.root, args.host, args.port, credential(TOKEN_ENV, TOKEN_SECRET), stop=stop)


if __name__ == "__main__":
    main()
