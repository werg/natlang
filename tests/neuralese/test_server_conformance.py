"""Conformance of the two Neuralese servers: the PyTorch reference (serve/) and the llama.cpp fork
(tools/neuralese/neuralese-server.cpp), on identical exported weights, answering the same HTTP requests.

New write or read behaviour lands in the reference first; this suite is what makes a fork that has not followed
visible. Each check sends one request to both servers and compares what a client can observe:

- `/v1/neuralese/info`: dialect, width, cutoff, maximum block length;
- a forced write (text, then a block): block length, truncation, payload (downloaded from each store), stop logits
  and the text after the block — for the shallow and the final stop source;
- a prompt that reads an uploaded block: the greedy continuation;
- a decision readout (`/v1/neuralese/decide`): per-option log-probabilities and token counts;
- text encoding (`/v1/neuralese/encode`, with and without a context), a write at a write site (`/write`) and the
  digest operator's plan (`/digest`: the fixture's single site, and a value chunked at a small window);
- capabilities the fork does not serve (gradient sessions, text embedding, weight adapters) answer with an error,
  not silence (an adapter request must never be answered by the base model).

Block IDs are content hashes of float payloads, so they differ whenever floats differ in the last bits; lengths
and payload closeness are compared instead. Skipped unless the fork's CPU build exists. The fork also runs as
WebAssembly (`final-wasm`: ts-host/vendor/neuralese-wasm through scripts/neuralese-wasm-server.mjs), the browser
runtime's service, against the same checks.
"""

from __future__ import annotations

import json
import shutil
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


def _wasm_command(model_gguf, heads_gguf):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = repo / "ts-host" / "scripts" / "neuralese-wasm-server.mjs"
    module = repo / "ts-host" / "vendor" / "neuralese-wasm" / "neuralese-wasm.wasm"
    node = shutil.which("node") or str(Path.home() / ".local" / "bin" / "node")
    if not module.exists() or not (repo / "ts-host" / "dist" / "browser" / "neuralese-wasm.js").exists():
        pytest.skip("the WebAssembly service or the ts-host build is missing")
    return [node, str(script), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "--max-block", "6"]


def _servers(loaded, tmp_path_factory, stop_source: str, impl: str = "native"):
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
    command = _wasm_command(model_gguf, heads_gguf) if impl == "wasm" else \
        [str(binary), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "-t", "8", "--max-block", "6"]
    fork = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    line = fork.stdout.readline()
    fork_url = json.loads(line)["listening"]
    host, port = reference.server_address[:2]
    return {"reference": f"http://{host}:{port}", "fork": fork_url, "process": fork, "server": reference, "engine": engine}


def heads_dialect():
    from natlang_neuralese.model.dialect import DIALECT

    return DIALECT


@pytest.fixture(scope="module", params=[("shallow", "native"), ("final", "native"), ("final", "wasm")],
                ids=["shallow", "final", "final-wasm"])
def servers(request, loaded, tmp_path_factory):
    pair = _servers(loaded, tmp_path_factory, *request.param)
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
    status, adapter = _json(servers["reference"] + "/v1/neuralese/adapters", "POST", {"kind": "xs", "rank": 2})
    assert status == 201
    bound = [{"id": adapter["id"], "scale": 1.0}]
    status, body = _json(servers["fork"] + "/v1/chat/completions", "POST",
                         {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "x_natlang_adapters": bound})
    assert status == 501 and body["error"]["code"] == "neuralese-adapters-unavailable"
    status, body = _json(servers["fork"] + "/v1/neuralese/decide", "POST",
                         {"messages": [{"role": "user", "content": "hi"}], "options": ["a", "b"], "adapters": bound})
    assert status == 501 and body["error"]["code"] == "neuralese-adapters-unavailable"


def _block_agrees(servers, got, atol=ATOL_PAYLOAD):
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 201, (ref, fork)
    assert ref["length"] == fork["length"] and bool(ref.get("truncated")) == bool(fork.get("truncated"))
    diff = (_payload(servers["reference"], ref["id"]) - _payload(servers["fork"], fork["id"])).abs().max().item()
    assert diff <= atol, f"payload differs by {diff:.3g}"
    return ref, fork


def test_encode_agrees(servers):
    for body in ({"text": "The plan: read the file, then count the fees.", "type": "Neuralese<string>"},
                 {"text": "count the fees", "context": [{"role": "user", "content": "Encode the instructions."}]}):
        ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/encode", "POST", body))
        assert ref.get("type") == fork.get("type")
        assert ref["producer"]["kind"] == fork["producer"]["kind"] == "text-encode"


def test_write_at_a_write_site_agrees(servers):
    body = {"messages": [{"role": "user", "content": "Summarise: the meeting moved to Tuesday."}], "prefix": "Note: "}
    _block_agrees(servers, _both(servers, "/v1/neuralese/write", "POST", body))


def test_digest_plans_agree(servers):
    """The fixture's write site (one write), and a long value at a small window (part writes and a combine write)."""
    from pathlib import Path

    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "digest-site.json").read_text())
    ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/digest", "POST", fixture["site"]))
    assert ref["parts"] == fork["parts"] == 1
    long = {**fixture["site"], "value": json.dumps({f"line{i}": f"fee {i} paid" for i in range(12)}), "window": 16}
    ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/digest", "POST", long), atol=3 * ATOL_PAYLOAD)
    assert ref["parts"] == fork["parts"] > 1 and ref["window"] == fork["window"] == 16


def test_template_readout_agrees(servers):
    """The reply forced to return_result cut from each server's own chat template: a written value, a decoded one."""
    opening = [{"role": "user", "content": "Combine the two notes into one value."}]
    template = {"call": "return_result", "arguments": {"status": "success"}, "value": "write"}
    got = _both(servers, "/v1/chat/completions", "POST", {"messages": opening, "max_tokens": 64, "neuralese_template": template})
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    calls = {name: got[name][1]["choices"][0]["message"].get("tool_calls") for name in got}
    for name, found in calls.items():
        assert found and found[0]["function"]["name"] == "return_result", (name, got[name][1])
        value = json.loads(found[0]["function"]["arguments"])["value"]
        assert value == [{"type": "neuralese", "id": got[name][1]["neuralese"]["blocks"][0]["id"]}], (name, value)
    rb, fb = ref["neuralese"]["blocks"][0], fork["neuralese"]["blocks"][0]
    assert rb["length"] == fb["length"]
    diff = (_payload(servers["reference"], rb["id"]) - _payload(servers["fork"], fb["id"])).abs().max().item()
    assert diff <= ATOL_PAYLOAD, f"payload differs by {diff:.3g}"
    assert ref["usage"]["completion_tokens"] == fork["usage"]["completion_tokens"]
    decode = {"messages": opening, "max_tokens": 12, "neuralese_template": {**template, "value": "decode"}}
    got = _both(servers, "/v1/chat/completions", "POST", decode)
    texts = {name: (got[name][0], got[name][1]["choices"][0]["message"], got[name][1]["usage"]["completion_tokens"]) for name in got}
    assert texts["reference"] == texts["fork"], texts


def test_length_hints_agree(servers):
    """A size hint (no stop decision), position by position and block-wise in one or all passes."""
    site = {"messages": [{"role": "user", "content": "Write the plan as a block."}], "prefix": "Plan: "}
    for hint in ({"length": 3}, {"length": 3, "passes": 1}, {"length": 3, "passes": 3}):
        ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/write", "POST", {**site, **hint}))
        assert ref["length"] == 3 and not ref.get("truncated")


def test_guidance_agrees(servers):
    """The reply checks agree on fixed replies, and a guided generation rolls back and retries the same way."""
    replies = [
        ("<|tool_call_start|>[eval(code='const s = state as any;\\nconst x = s.a.filter((l:any)=>l.amount>0);\\n"
         "if (x.length) {\\n  return x;\\n}\\nreturn [];')]<|tool_call_end|>", {}),
        ("<|tool_call_start|>[eval(code='const s = state as any;\\nconst x = s.a.filter((l:any)=>l.amount>0;\\nreturn x;')]", {}),
        ("<|tool_call_start|>[eval(code='const a = 1;\\n" + "const positiveLines = positiveLines.filter(f);\\n" * 4, {"repeat": 3}),
        ("<|tool_call_start|>[evaluate(code='1')]", {"tools": ["eval"]}),
        ("<|tool_call_start|>[eval(code='const t = `a ${b}`;\\nconst o = {a: 1, b: [2, 3]};\\nawait f(o)\\n')]", {}),
    ]
    verdicts = []
    for reply, guidance in replies:
        got = _both(servers, "/v1/neuralese/guidance/check", "POST", {"reply": reply, "guidance": guidance})
        assert got["reference"][1] == got["fork"][1], (reply, got)
        verdicts.append(got["reference"][1]["reason"])
    assert verdicts == [None, "syntax", "repetition", "unknown-tool", None]
    tools = [{"type": "function", "function": {"name": "eval", "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}}}}}, {"type": "function", "function": {"name": "return_result", "parameters": {}}}]
    body = {"messages": [{"role": "user", "content": "Count the fees."}], "max_tokens": 16, "tools": tools,
            "tool_choice": "required", "guidance": {"retries": 2, "tools": ["return_result"]}}
    got = _both(servers, "/v1/chat/completions", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    assert ref["x_natlang_guidance"] == fork["x_natlang_guidance"] and ref["x_natlang_guidance"]["rejections"]
    assert ref["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == \
        fork["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "return_result"


def test_rendered_prompts_agree(servers):
    """Both servers render the same prompt: plain messages, tools, tool calls with results."""
    tools = [{"type": "function", "function": {"name": "eval", "description": "Run TypeScript.", "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
    for body in ({"messages": [{"role": "user", "content": "hi"}]},
                 {"messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Count the fees."}], "tools": tools},
                 {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [
                     {"id": "c0", "type": "function", "function": {"name": "eval", "arguments": "{\"code\": \"return 1;\"}"}}]},
                     {"role": "tool", "tool_call_id": "c0", "content": "1"}], "tools": tools}):
        got = _both(servers, "/v1/neuralese/render", "POST", body)
        assert got["reference"][1]["prompt"] == got["fork"][1]["prompt"], (got["reference"][1]["prompt"], got["fork"][1]["prompt"])
