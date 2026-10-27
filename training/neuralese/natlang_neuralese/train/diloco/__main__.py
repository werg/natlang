"""Serve a DiLoCo coordinator root, or print its status.

    python -m natlang_neuralese.train.diloco serve --root R [--host 0.0.0.0] [--port 8080]
    python -m natlang_neuralese.train.diloco status --root R

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--root", type=Path, required=True)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    status = commands.add_parser("status")
    status.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "status":
        print(json.dumps(Coordinator(args.root).status(), indent=2, sort_keys=True))
        return
    stop = threading.Event()
    # SIGTERM stops serving and merges what learners sent last (Coordinator.close).
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    serve_coordinator(args.root, args.host, args.port, credential(TOKEN_ENV, TOKEN_SECRET), stop=stop)


if __name__ == "__main__":
    main()
