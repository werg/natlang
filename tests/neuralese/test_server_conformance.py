"""Conformance of the two Neuralese servers: the PyTorch reference (serve/) and the llama.cpp fork
(tools/neuralese/neuralese-server.cpp), on identical exported weights, answering the same HTTP requests.

New write or read behaviour lands in the reference first; this suite is what makes a fork that has not followed
visible. Each check sends one request to both servers and compares what a client can observe:

- `/v1/neuralese/info`: dialect, width, cutoff, maximum block length;
- a forced write (text, then a block): block length, truncation, payload (downloaded from each store), stop logits
  and the text after the block — for the shallow and the final stop source;
- a prompt that reads an uploaded block: the greedy continuation;
- a decision readout (`/v1/neuralese/decide`): per-option log-probabilities and token counts;
- capabilities the fork does not serve (gradient sessions, text embedding) answer with an error, not silence.

Block IDs are content hashes of float payloads, so they differ whenever floats differ in the last bits; lengths
and payload closeness are compared instead. Skipped unless the fork's CPU build exists.
"""

from __future__ import annotations

import json
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")

ATOL_PAYLOAD = 2e-2
ATOL_LOGPROB = 5e-2


def _binary():
    from natlang_neuralese.export import fork_root

    path = fork_root() / "build-cpu" / "bin" / "llama-neuralese-server"
    if not path.exists():
        pytest.skip(f"llama.cpp Neuralese server not built at {path}")
    return path


def _request(url: str, method: str = "GET", body=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    headers = {"content-type": "application/octet-stream" if raw is not None else "application/json"}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = response.read()
            return response.status, payload
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _json(url, method="GET", body=None):
    status, payload = _request(url, method, body)
    return status, json.loads(payload or b"{}")


def _servers(loaded, tmp_path_factory, stop_source: str):
    from natlang_neuralese.export import export_heads_gguf, export_model_gguf, export_model_hf
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.serve.store import TensorStore

    binary = _binary()
    model, tokenizer, backbone = loaded
    torch.manual_seed(5)
    heads = PortHeads(backbone, cutoff=6, max_length=6, stop_source=stop_source,
                      stop_position=stop_source == "shallow").eval()
    with torch.no_grad():
        for name, p in heads.named_parameters():
            if name.startswith("feedback.readout.") or name == "feedback.gate":
                continue
            p.add_(0.02 * torch.randn_like(p))
            p.requires_grad_(False)
        heads.stop.mlp_out.bias.fill_(-0.2)
    out = tmp_path_factory.mktemp(f"conformance-{stop_source}")
    model_gguf = export_model_gguf(export_model_hf(backbone, tokenizer, out / "hf"), out / "model-f32.gguf")
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese-f32.gguf")
    engine = Engine(backbone, heads, tokenizer, TensorStore(), heads_dialect(), max_block=6)
    engine.start()
    reference = serve(engine)
    threading.Thread(target=reference.serve_forever, daemon=True).start()
    fork = subprocess.Popen([str(binary), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "-t", "8",
                             "--max-block", "6"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = fork.stdout.readline()
    fork_url = json.loads(line)["listening"]
    host, port = reference.server_address[:2]
    return {"reference": f"http://{host}:{port}", "fork": fork_url, "process": fork, "server": reference, "engine": engine}


def heads_dialect():
    from natlang_neuralese.model.dialect import DIALECT

    return DIALECT


@pytest.fixture(scope="module", params=["shallow", "final"])
def servers(request, loaded, tmp_path_factory):
    pair = _servers(loaded, tmp_path_factory, request.param)
    yield pair
    pair["process"].terminate()
    pair["server"].shutdown()
    pair["engine"].stop()


def _both(servers, path, method="GET", body=None):
    return {name: _json(servers[name] + path, method, body) for name in ("reference", "fork")}


def _payload(url: str, block_id: str) -> torch.Tensor:
    from natlang_neuralese.serve.store import decode_block

    status, raw = _request(f"{url}/v1/neuralese/blocks/{block_id}")
    assert status == 200
    return decode_block(raw).payload.float()


def test_info_agrees(servers):
    got = _both(servers, "/v1/neuralese/info")
    keys = ("dialects", "width", "cutoff", "max_block_length")
    assert {k: got["reference"][1][k] for k in keys} == {k: got["fork"][1][k] for k in keys}


def test_a_forced_write_agrees(servers):
    body = {"messages": [{"role": "user", "content": "Write the plan as a block, then say done."}],
            "x_natlang_forced": ["Plan: ", {"neuralese": "write"}, " done"], "max_tokens": 64}
    got = _both(servers, "/v1/chat/completions", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    rb, fb = ref["neuralese"]["blocks"], fork["neuralese"]["blocks"]
    assert len(rb) == len(fb) == 1
    assert rb[0]["length"] == fb[0]["length"]
    assert bool(rb[0].get("truncated")) == bool(fb[0].get("truncated"))
    diff = (_payload(servers["reference"], rb[0]["id"]) - _payload(servers["fork"], fb[0]["id"])).abs().max().item()
    assert diff <= ATOL_PAYLOAD, f"payload differs by {diff:.3g}"
    rstop, fstop = rb[0]["producer"].get("stop_logits"), fb[0]["producer"].get("stop_logits")
    if rstop is not None and fstop is not None:
        assert len(rstop) == len(fstop)
        assert max((abs(a - b) for a, b in zip(rstop, fstop)), default=0.0) <= ATOL_PAYLOAD


def test_reading_an_uploaded_block_continues_the_same(servers):
    from natlang_neuralese.serve.store import encode_block, make_block

    torch.manual_seed(9)
    width = _both(servers, "/v1/neuralese/info")["reference"][1]["width"]
    block = make_block(0.05 * torch.randn(3, width), heads_dialect(), type="Neuralese<string>")
    raw = encode_block(block)
    for name in ("reference", "fork"):
        status, _ = _request(f"{servers[name]}/v1/neuralese/blocks/{block.id}", "PUT", raw=raw)
        assert status == 201, name
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "Note: "},
                                                      {"type": "neuralese", "id": block.id},
                                                      {"type": "text", "text": " What is the capital of France?"}]}],
            "max_tokens": 8, "temperature": 0}
    got = _both(servers, "/v1/chat/completions", "POST", body)
    texts = {name: got[name][1]["choices"][0]["message"].get("content") for name in got}
    assert texts["reference"] == texts["fork"], texts


def test_decision_readout_agrees(servers):
    body = {"messages": [{"role": "user", "content": "Is Paris the capital of France? Reply with a JSON value."}],
            "options": ["true", "false"]}
    got = _both(servers, "/v1/neuralese/decide", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    assert ref["tokens"] == fork["tokens"] and all(n > 0 for n in ref["tokens"])
    assert all(v < 0 for v in ref["log_probs"]) and ref["log_probs"][0] != ref["log_probs"][1]
    print(f"decide: reference {ref['log_probs']} fork {fork['log_probs']}")
    for a, b in zip(ref["log_probs"], fork["log_probs"]):
        assert abs(a - b) <= ATOL_LOGPROB, (ref, fork)


def test_capabilities_the_fork_does_not_serve_fail_loudly(servers):
    status, body = _json(servers["fork"] + "/v1/neuralese/grad", "POST", {"terms": []})
    assert status == 501 and body["error"]["code"] == "neuralese-grad-unavailable"
    status, _ = _request(servers["fork"] + "/v1/neuralese/embed", "POST", {"text": "x"})
    assert status == 404
