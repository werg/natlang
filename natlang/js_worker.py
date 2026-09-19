"""Private JSON-lines bridge for the effectful QuickJS worker."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.js import _execute, EffectError
from natlang.diag import Reject


def send(message):
    print(json.dumps(message), flush=True)


def effect(cap, fn, args):
    send({"kind": "effect", "cap": cap, "fn": fn, "args": args})
    reply = json.loads(sys.stdin.readline())
    if "__error" in reply:
        raise EffectError(reply["__error"])
    return reply.get("value")


def main():
    request = json.loads(sys.stdin.readline())
    try:
        value = _execute(request["code"], request["scope"], effect, body=request["body"],
                         path=request["path"], effectful=True)
        send({"kind": "result", "value": value})
    except Reject as e:
        send({"kind": "reject", "diags": [asdict(d) for d in e.diags]})
    except Exception as e:
        send({"kind": "error", "message": str(e)})


if __name__ == "__main__":
    main()
