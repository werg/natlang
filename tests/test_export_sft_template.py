"""The optional text renderer must preserve the server's template identity."""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class TemplateServer(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        self._send({"chat_template": "fixture-chatml-v1"})

    def do_POST(self):
        size = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(size))
        parts = []
        for message in request["messages"]:
            body = message.get("content", "")
            if message.get("tool_calls"):
                body = json.dumps(message["tool_calls"])
            parts.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>")
        prompt = "".join(parts)
        if request["messages"][-1]["role"] != "assistant":
            prompt += "<|im_start|>assistant\n"
        self._send({"prompt": prompt})

    def _send(self, value):
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def test_template_export_records_identity_and_rejects_changed_resume(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), TemplateServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        src, dst = tmp_path / "trace.jsonl", tmp_path / "sft.jsonl"
        row = {"id": "one", "program_id": "p", "family": "fixture",
               "skill": "write", "messages": [
                   {"role": "system", "content": "rules"},
                   {"role": "user", "content": "question"}],
               "tools": [], "target": {"role": "assistant", "content": "",
                   "tool_calls": [{"type": "function", "function": {
                       "name": "write", "arguments": "{}"}}]}}
        src.write_text(json.dumps(row) + "\n" +
                       json.dumps({**row, "id": "provisional", "provisional_gold": True}) + "\n")
        repo = Path(__file__).resolve().parents[1]
        base = [sys.executable, str(repo / "scripts/export_sft.py"),
                str(src), str(dst), "--server", f"http://127.0.0.1:{server.server_port}"]
        result = subprocess.run(base + ["--template-id", "fixture-v1"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        sample = json.loads(dst.read_text())
        assert sample["id"] == "one"  # a template leaf taints its whole program
        assert sample["completion"].endswith("<|im_end|>")
        manifest = json.loads((tmp_path / "sft.jsonl.manifest.json").read_text())
        assert manifest["renderer"]["template_sha256"]
        changed = subprocess.run(base + ["--template-id", "fixture-v2", "--resume"],
                                 capture_output=True, text=True)
        assert changed.returncode != 0
        assert "renderer differs" in changed.stderr
        provisional_dst = tmp_path / "provisional-sft.jsonl"
        included = subprocess.run([*base[:3], str(provisional_dst), *base[4:],
                                   "--include-template"], capture_output=True, text=True)
        assert included.returncode == 0, included.stderr
        assert len(provisional_dst.read_text().splitlines()) == 2
        grouped_src, grouped_dst = tmp_path / "grouped.jsonl", tmp_path / "grouped-sft.jsonl"
        scenario = {**row, "id": "scenario-turn", "family": "lambda_scenario",
                    "source_groups": ["scenario:support:113:0"]}
        review = {**row, "id": "review-turn", "kind": "review",
                  "source_groups": ["scenario:support:113:0"]}
        review.pop("family")
        grouped_src.write_text(json.dumps(scenario) + "\n" + json.dumps(review) + "\n")
        grouped = subprocess.run([base[0], base[1], str(grouped_src), str(grouped_dst),
                                  *base[4:]], capture_output=True, text=True)
        assert grouped.returncode == 0, grouped.stderr
        pairs = [json.loads(line) for line in grouped_dst.read_text().splitlines()]
        assert [p["program_id"] for p in pairs] == ["scenario:support:113:0"] * 2
        assert pairs[1]["family"] == "review"
        reply_src, reply_dst = tmp_path / "reply.jsonl", tmp_path / "reply-sft.jsonl"
        reply_src.write_text(json.dumps({**row, "id": "reply", "skill": "reply", "tools": [],
                                         "target": {"role": "assistant", "content": "Done."}}) + "\n")
        exported = subprocess.run([base[0], base[1], str(reply_src), str(reply_dst),
                                   *base[4:]], capture_output=True, text=True)
        assert exported.returncode == 0, exported.stderr
        assert json.loads(reply_dst.read_text())["completion"] == "<|im_end|>"
    finally:
        server.shutdown()
        thread.join()
