"""Parity of the llama.cpp fork's direct-v1 Neuralese runtime with the PyTorch reference server on a direct artifact.

The reference is `natlang_neuralese.serve --direct-artifact` (train/direct_export.load_direct_artifact, the
DirectWriter of serve/direct_write.py); the fork serves the same artifact exported by
`python -m natlang_neuralese.export --direct-artifact` (model GGUF with the trained backbone, direct-v1 projector
GGUF) through tools/neuralese (nz_write_direct). Both answer the same HTTP requests on CPU in FP32:

- info: width, cutoff (full depth), maximum block length;
- forced writes: block length, truncation, the soft-token values (payload) and the stop log-odds per step;
- a write at a write site with no length (the readout's stop tokens close it) and with a length hint;
- reads: a prompt holding a written block, its greedy continuation and decision log-probabilities (next-token
  readout after the raw-identity read).

Needs NATLANG_DIRECT_ARTIFACT and NATLANG_DIRECT_BASE_SNAPSHOT (e.g. the LFM2.5-350M step-10006 export and its pinned
base snapshot) and the fork's CPU build; skipped otherwise. Run under the memory ledger.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")

ARTIFACT = os.environ.get("NATLANG_DIRECT_ARTIFACT")
BASE = os.environ.get("NATLANG_DIRECT_BASE_SNAPSHOT")
pytestmark = pytest.mark.skipif(not (ARTIFACT and BASE), reason="NATLANG_DIRECT_ARTIFACT/NATLANG_DIRECT_BASE_SNAPSHOT unset")

ATOL_PAYLOAD = 2e-2
ATOL_LOGPROB = 5e-2
THREADS = int(os.environ.get("NATLANG_CONFORMANCE_THREADS") or min(8, os.cpu_count() or 1))
CONTEXT = 4096
MAX_BLOCK = 24
REPORT: dict = {}


def _request(url, method="GET", body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"content-type": "application/octet-stream" if raw is not None else "application/json"}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _json(url, method="GET", body=None):
    status, raw = _request(url, method, body)
    return status, json.loads(raw) if raw else None


@pytest.fixture(scope="module")
def servers(tmp_path_factory):
    from natlang_neuralese.export import export_direct_heads_gguf, export_model_gguf, export_model_hf, fork_root
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.train.direct_export import load_direct_artifact

    binary = fork_root() / "build-cpu" / "bin" / "llama-neuralese-server"
    if not binary.exists():
        pytest.skip(f"llama.cpp Neuralese server not built at {binary}")
    torch.set_num_threads(THREADS)
    engine = load_direct_artifact(ARTIFACT, base_snapshot=BASE, device="cpu", dtype=torch.float32).engine
    engine.max_block = min(engine.max_block, MAX_BLOCK)
    engine.context = min(engine.context, CONTEXT)
    out = tmp_path_factory.mktemp("direct-parity")
    model_gguf = export_model_gguf(export_model_hf(engine.backbone, engine.tokenizer, out / "hf"), out / "model-f32.gguf")
    heads_gguf = export_direct_heads_gguf(engine, out / "neuralese-direct-f32.gguf")
    engine.start()
    reference = serve(engine)
    threading.Thread(target=reference.serve_forever, daemon=True).start()
    command = [str(binary), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "-t", str(THREADS),
               "--max-block", str(engine.max_block), "-c", str(CONTEXT)]
    fork = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    line = fork.stdout.readline()
    if not line:
        raise RuntimeError("fork server did not start: " + fork.stderr.read()[-4000:])
    host, port = reference.server_address[:2]
    pair = {"reference": f"http://{host}:{port}", "fork": json.loads(line)["listening"], "engine": engine}
    yield pair
    fork.terminate()
    reference.shutdown()
    engine.stop()
    path = os.environ.get("NATLANG_DIRECT_PARITY_REPORT")
    if path:
        with open(path, "w") as handle:
            json.dump(REPORT, handle, indent=1)


def _both(servers, path, method="GET", body=None):
    return {name: _json(servers[name] + path, method, body) for name in ("reference", "fork")}


def _payload(url, block_id):
    from natlang_neuralese.serve.store import decode_block

    status, raw = _request(f"{url}/v1/neuralese/blocks/{block_id}")
    assert status == 200
    return decode_block(raw).payload.float()


def _blocks_agree(servers, name, rb, fb):
    assert rb["length"] == fb["length"], (name, rb["length"], fb["length"])
    assert bool(rb.get("truncated")) == bool(fb.get("truncated")), name
    ref, fork = _payload(servers["reference"], rb["id"]), _payload(servers["fork"], fb["id"])
    diff = (ref - fork).abs()
    scale = ref.abs().max().item()
    rstop, fstop = rb["producer"].get("stop_logits") or [], fb["producer"].get("stop_logits") or []
    assert len(rstop) == len(fstop), (name, rstop, fstop)
    stop_diff = max((abs(a - b) for a, b in zip(rstop, fstop)), default=0.0)
    REPORT[name] = {"length": rb["length"], "truncated": bool(rb.get("truncated")), "payload_max_abs_diff": diff.max().item(),
                    "payload_mean_abs_diff": diff.mean().item(), "payload_max_abs": scale, "stop_logits_max_diff": stop_diff,
                    "fork_algorithm": fb["producer"].get("algorithm"), "reference_algorithm": rb["producer"].get("algorithm")}
    print(name, json.dumps(REPORT[name]))
    assert diff.max().item() <= ATOL_PAYLOAD, f"{name}: payload differs by {diff.max().item():.3g}"
    assert stop_diff <= ATOL_PAYLOAD, f"{name}: stop log-odds differ by {stop_diff:.3g}"
    assert fb["producer"].get("algorithm") == rb["producer"].get("algorithm") == "neuralese-direct-v1"


def test_info_agrees(servers):
    got = _both(servers, "/v1/neuralese/info")
    keys = ("dialects", "width", "cutoff", "max_block_length")
    assert {k: got["reference"][1][k] for k in keys} == {k: got["fork"][1][k] for k in keys}


@pytest.mark.parametrize("case", [
    ("forced-plan", ["Plan: ", {"neuralese": "write"}, " done"], "Write the plan as a block, then say done."),
    ("forced-code", ["The value is ", {"neuralese": "write"}], "Compute the sum of the first ten squares and give it as a block."),
])
def test_forced_writes_agree(servers, case):
    name, forced, prompt = case
    body = {"messages": [{"role": "user", "content": prompt}], "x_natlang_forced": forced, "max_tokens": 64}
    got = _both(servers, "/v1/chat/completions", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    rb, fb = ref["neuralese"]["blocks"], fork["neuralese"]["blocks"]
    assert len(rb) == len(fb) == 1
    _blocks_agree(servers, name, rb[0], fb[0])
    assert ref["usage"] == fork["usage"], (ref["usage"], fork["usage"])


@pytest.mark.parametrize("hint", [None, 4])
def test_writes_at_a_write_site_agree(servers, hint):
    site = {"messages": [{"role": "user", "content": "Summarise: the cat sat on the mat. Write the summary as a block."}],
            "prefix": "Summary: "}
    if hint:
        site["length"] = hint
    got = _both(servers, "/v1/neuralese/write", "POST", site)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 201, (ref, fork)
    _blocks_agree(servers, f"write-site-{hint or 'free'}", ref, fork)
    if hint:
        assert ref["length"] == hint


def test_reading_a_written_block_agrees(servers):
    """A block written by the reference, uploaded to both: next-token readout after the raw-identity read."""
    from natlang_neuralese.serve.store import decode_block, encode_block

    site = {"messages": [{"role": "user", "content": "Name a colour and write it as a block."}], "prefix": "Colour: "}
    status, written = _json(servers["reference"] + "/v1/neuralese/write", "POST", site)
    assert status == 201
    status, raw = _request(f"{servers['reference']}/v1/neuralese/blocks/{written['id']}")
    assert status == 200
    block = decode_block(raw)
    assert _request(f"{servers['fork']}/v1/neuralese/blocks/{block.id}", "PUT", raw=encode_block(block))[0] in (200, 201)
    content = [{"type": "text", "text": "Here is a value: "}, {"type": "neuralese", "id": block.id},
               {"type": "text", "text": " Is it a colour? Reply with a JSON value."}]
    decide = _both(servers, "/v1/neuralese/decide", "POST",
                   {"messages": [{"role": "user", "content": content}], "options": ["true", "false"]})
    (rs, ref), (fs, fork) = decide["reference"], decide["fork"]
    assert rs == fs == 200, (ref, fork)
    assert ref["tokens"] == fork["tokens"]
    diff = max(abs(a - b) for a, b in zip(ref["log_probs"], fork["log_probs"]))
    reply = _both(servers, "/v1/chat/completions", "POST",
                  {"messages": [{"role": "user", "content": content}], "max_tokens": 12, "temperature": 0})
    texts = {name: reply[name][1]["choices"][0]["message"].get("content") for name in reply}
    REPORT["read"] = {"block_length": block.payload.shape[0], "decide_log_probs": {"reference": ref["log_probs"],
                      "fork": fork["log_probs"]}, "decide_max_diff": diff, "greedy": texts}
    print("read", json.dumps(REPORT["read"]))
    assert diff <= ATOL_LOGPROB, (ref, fork)
    assert texts["reference"] == texts["fork"], texts
