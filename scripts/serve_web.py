#!/usr/bin/env python3
"""Host adapter for codebases/webserver: a real listening HTTP server whose every request is an event of a
natlang fold. The HTTP thread puts the request on a queue and waits; the natlang run pulls it from the open list,
interprets handle.nl, and the `http.respond` capability hands the response back to the waiting thread.

  scripts/serve_web.py --port 8000 --server http://127.0.0.1:8081 --thinking 256 --alias call=call_function
The model is the interpreter of every step. Stop with Ctrl-C (the fold is closed and the final state is printed).
"""
import argparse, json, queue, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.decoder import LlamaServerDecoder
from natlang.host import load_fold
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import dump


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--server", default="http://127.0.0.1:8080", help="llama.cpp server of the interpreter model")
    ap.add_argument("--decode", default="native", choices=("native", "server"))
    ap.add_argument("--thinking", type=int, default=None)
    ap.add_argument("--alias", action="append", default=[])
    ap.add_argument("--system-file", type=Path, default=ROOT / "natlang" / "prompts" / "tools_delegate.md")
    ap.add_argument("--site", type=Path, default=ROOT / "codebases" / "webserver" / "site.yaml")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    events, waiting, counter = queue.Queue(), {}, iter(range(1, 10**9))

    class Handler(BaseHTTPRequestHandler):
        def _handle(self):
            rid = f"r{next(counter)}"
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode("utf-8", "replace")
            done = waiting[rid] = queue.Queue(maxsize=1)
            events.put({"id": rid, "method": self.command, "path": self.path, "headers": dict(self.headers.items()), "body": body})
            try:
                resp = done.get(timeout=a.timeout)
            except queue.Empty:
                resp = {"status": 504, "headers": {"Content-Type": "text/plain"}, "body": "the interpreter did not answer in time"}
            data = resp["body"].encode()
            self.send_response(int(resp["status"]))
            for k, v in resp["headers"].items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        do_GET = do_POST = do_PUT = do_DELETE = _handle

    def source():
        while True:
            e = events.get()
            if e == "$close":
                return
            yield e

    extra = {} if a.thinking is None else {"thinking_budget_tokens": a.thinking, "top_p": 0.95, "top_k": 20}
    dec = (NativeCallDecoder(a.server, timeout=a.timeout) if a.decode == "native" else
           LlamaServerDecoder(a.server, timeout=a.timeout, chat_extra=extra, tool_aliases=dict(x.split("=", 1) for x in a.alias)))
    prompt = a.system_file.read_text()
    rt = Runtime(lambda lam: ToolAgent(dec, temperature=0.3, system_prompt=prompt), max_episodes=10**6,
                 capabilities={"http.respond": lambda args: waiting.pop(args[0]).put(args[1])})
    root = load_fold(ROOT / "codebases" / "webserver" / "handle.nl", yaml.safe_load(a.site.read_text()), source())
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"natlang web server on http://127.0.0.1:{a.port}  (interpreter: {a.server}, {a.decode})", flush=True)
    result = {}
    runner = threading.Thread(target=lambda: result.update(zip(("out", "value"), rt.run_root(root))), daemon=True)
    runner.start()
    try:
        runner.join()
    except KeyboardInterrupt:
        events.put("$close")
        runner.join(timeout=5)
    httpd.shutdown()
    out = result.get("out")
    print(f"# {out.kind if out else 'interrupted'}" + (f": {out.detail}" if out and out.kind != "done" else ""))
    if out and out.kind == "done":
        print(json.dumps(dump(result["value"]).get("log"), indent=1))


if __name__ == "__main__":
    main()
