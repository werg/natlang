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
  Neuralese instance of the builtin `view` (`/view`: the fixture's single site, faithful and instructed, and a value
  chunked at a small window);
- weight adapters: an `xs` adapter exported as a GGUF LoRA (export/adapters.py) and loaded into the fork gives the
  reference's decision log-probabilities and greedy reply;
- capabilities the fork does not serve (gradient sessions, optimiser steps, adapter creation and export, adapters it
  has no LoRA for or decoded through a projection) answer with an error, not silence (an adapter request must never be
  answered by the base model);
- fork parity: text embedding, template value types and argument paths, parts typed "unknown", Qwen-style calls and
  thinking, info fields, batched item errors, the width check on read, and streamed replies (each assembled equals the
  non-streamed reply, and the two servers stream the same).

Block IDs are content hashes of float payloads, so they differ whenever floats differ in the last bits; lengths
and payload closeness are compared instead. Skipped unless the fork's CPU build exists. The fork also runs as
WebAssembly (`final-wasm`: ts-host/vendor/neuralese-wasm through scripts/neuralese-wasm-server.mjs), the browser
runtime's service, against the same checks; `final-wasm-mt` the threaded build with four threads (NATLANG_CONFORMANCE_WASM_MT=0
leaves it out). The WebGPU build needs a browser's WebGPU and is not run under Node. Both servers compute on
NATLANG_CONFORMANCE_THREADS CPU threads (default min(8, cores); `THREADS`).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

torch = pytest.importorskip("torch")

ATOL_PAYLOAD = 2e-2
ATOL_LOGPROB = 5e-2
# The served context of every pair (the fork's -c, the reference Engine's context): view plans its write sites against it.
CONTEXT = 8192
# CPU threads of both servers: the reference's torch intra-op pool (torch.set_num_threads, process-wide, so also the
# engine's and the HTTP handlers' threads) and the native fork's -t. NATLANG_CONFORMANCE_THREADS, default min(8, cores):
# torch's default (every core) crawls when the machine is loaded. The threaded wasm build keeps its own four threads.
THREADS = int(os.environ.get("NATLANG_CONFORMANCE_THREADS") or min(8, os.cpu_count() or 1))


@pytest.fixture(scope="module", autouse=True)
def _cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(THREADS)
    yield THREADS
    torch.set_num_threads(before)


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


def _wasm_command(model_gguf, heads_gguf, build: str = "neuralese-wasm"):
    """The vendored WebAssembly service under Node: `neuralese-wasm` (one thread) or `neuralese-wasm-mt` (pthreads,
    four threads). The WebGPU build needs a browser's WebGPU (Chromium, JSPI); Node has none, so it is measured with
    scripts/browser-neuralese-pilot.mjs instead."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    script = repo / "ts-host" / "scripts" / "neuralese-wasm-server.mjs"
    # NATLANG_CONFORMANCE_WASM_DIR: a directory holding the builds to check before they are vendored (e.g. a fork build's
    # bin directory); default the vendored ones.
    vendor = os.environ.get("NATLANG_CONFORMANCE_WASM_DIR")
    module = (Path(vendor) if vendor else repo / "ts-host" / "vendor" / "neuralese-wasm") / f"{build}.wasm"
    node = shutil.which("node") or str(Path.home() / ".local" / "bin" / "node")
    if not module.exists() or not (repo / "ts-host" / "dist" / "browser" / "neuralese-wasm.js").exists():
        pytest.skip("the WebAssembly service or the ts-host build is missing")
    threads = ["--module", str(module.with_suffix(".mjs"))] + (["-t", "4"] if build == "neuralese-wasm-mt" else [])
    return [node, str(script), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "--max-block", "6",
            "-c", str(CONTEXT), *threads]


def _servers(loaded, tmp_path_factory, stop_source: str, impl: str = "native"):
    from natlang_neuralese.export import export_heads_gguf, export_model_gguf, export_model_hf
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.serve.store import TensorStore

    binary = _binary()
    if stop_source == "trained":
        # A trained port checkpoint (NATLANG_CONFORMANCE_HEADS): its heads and any phase-F deltas on the backbone.
        from natlang_neuralese.serve import load_engine

        engine = load_engine(heads_checkpoint=os.environ["NATLANG_CONFORMANCE_HEADS"], device="cpu",
                             dtype=torch.float32)
        backbone, heads, tokenizer = engine.backbone, engine.heads, engine.tokenizer
    else:
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
        engine = None
    out = tmp_path_factory.mktemp(f"conformance-{stop_source}")
    model_gguf = export_model_gguf(export_model_hf(backbone, tokenizer, out / "hf"), out / "model-f32.gguf")
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese-f32.gguf")
    engine = engine or Engine(backbone, heads, tokenizer, TensorStore(), heads_dialect(), max_block=6, context=CONTEXT)
    engine.context = min(engine.context, CONTEXT)
    engine.start()
    reference = serve(engine)
    threading.Thread(target=reference.serve_forever, daemon=True).start()
    command = _wasm_command(model_gguf, heads_gguf) if impl == "wasm" else \
        _wasm_command(model_gguf, heads_gguf, "neuralese-wasm-mt") if impl == "wasm-mt" else \
        [str(binary), "-m", str(model_gguf), "--nz", str(heads_gguf), "--port", "0", "-t", str(THREADS), "--max-block", str(engine.max_block),
         "-c", str(CONTEXT)]
    fork, fork_url = _start(command)
    host, port = reference.server_address[:2]
    return {"reference": f"http://{host}:{port}", "fork": fork_url, "process": fork, "server": reference, "engine": engine,
            "hf": out / "hf", "command": command}


def _start(command):
    fork = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return fork, json.loads(fork.stdout.readline())["listening"]


def heads_dialect():
    from natlang_neuralese.model.dialect import DIALECT

    return DIALECT


# NATLANG_CONFORMANCE_HEADS=checkpoint.pt adds a pair serving a trained port (S3 pilot or full run).
_TRAINED = [("trained", "native")] if os.environ.get("NATLANG_CONFORMANCE_HEADS") else []
# The threaded WebAssembly build under Node (`final-wasm-mt`, four threads) runs by default; NATLANG_CONFORMANCE_WASM_MT=0
# leaves it out. Until fork 8b41aecff it deadlocked intermittently (ggml created and joined its worker threads per graph,
# and one worker hung in emscripten's thread exit while the main thread joined it); the service now keeps one persistent
# threadpool.
_WASM_MT = [] if os.environ.get("NATLANG_CONFORMANCE_WASM_MT") == "0" else [("final", "wasm-mt")]


@pytest.fixture(scope="module", params=[("shallow", "native"), ("final", "native"), ("final", "wasm")] + _WASM_MT + _TRAINED,
                ids=["shallow", "final", "final-wasm"] + ["final-wasm-mt"] * len(_WASM_MT) + ["trained"] * len(_TRAINED))
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


def test_decision_after_an_assistant_turn_with_reasoning_scores_the_generation_prompt(servers):
    """A decision whose prompt holds an earlier assistant turn with reasoning (the runtime's `scope_` turn): LFM2.5's
    template (history policy last_turn_only) keeps that reasoning while the turn is the last assistant turn, as when the
    model generates the reply, and drops it once the scored option follows as another assistant turn. Both servers
    score the option after the prompt as generation renders it (reasoning kept), cutting the option from the full
    rendering after its own generation prefix; neither refuses it, and the scores agree. A server that scored the
    full rendering's prefix would score the variant without the reasoning."""
    tokenizer = servers["engine"].tokenizer
    turn = {"role": "assistant", "content": "Scoped.", "reasoning_content": "Only the capital matters here."}
    messages = [{"role": "user", "content": "Is Paris the capital of France?"}, turn,
                {"role": "user", "content": "Reply with a JSON value."}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": "true"}], tokenize=False)
    assert turn["reasoning_content"] in prompt and turn["reasoning_content"] not in full, "the template keeps reasoning"
    body = {"messages": messages, "options": ["true", "false"]}
    got = _both(servers, "/v1/neuralese/decide", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    assert ref["tokens"] == fork["tokens"]
    for a, b in zip(ref["log_probs"], fork["log_probs"]):
        assert abs(a - b) <= ATOL_LOGPROB, (ref, fork)
    plain = dict(turn)
    plain.pop("reasoning_content")
    without = _both(servers, "/v1/neuralese/decide", "POST", {**body, "messages": [messages[0], plain, messages[2]]})
    for name in ("reference", "fork"):
        assert without[name][0] == 200
        assert max(abs(a - b) for a, b in zip(got[name][1]["log_probs"], without[name][1]["log_probs"])) > 1e-4, name
    many = _both(servers, "/v1/natlang/score", "POST", {"items": [{"messages": messages, "continuations": ["true", "false"]}]})
    for name in ("reference", "fork"):
        item = many[name][1]["results"][0]
        assert "error" not in item, (name, item)
        assert max(abs(a - b) for a, b in zip(item["log_probs"], got[name][1]["log_probs"])) <= 1e-4, name


def test_batched_decision_scoring_agrees_and_fails_per_item(servers):
    """POST /v1/natlang/score (plans/BATCHED_EXECUTION.md): each item equals /v1/neuralese/decide alone; items
    with the same prompt share one prefill; a bad item fails alone; /v1/neuralese/decide_many is the same."""
    france = [{"role": "user", "content": "Is Paris the capital of France? Reply with a JSON value."}]
    water = [{"role": "user", "content": "Is water wet? Reply with a JSON value."}]
    items = [{"messages": france, "continuations": ["true", "false"]},
             {"messages": france, "continuations": ["\"yes\"", "\"no\"", "\"maybe\""]},
             {"messages": water, "continuations": ["true", "false"]},
             {"messages": france, "continuations": []}]
    for path in ("/v1/natlang/score", "/v1/neuralese/decide_many"):
        got = _both(servers, path, "POST", {"items": items})
        (rs, ref), (fs, fork) = got["reference"], got["fork"]
        assert rs == fs == 200, (ref, fork)
        assert "error" in ref["results"][3] and "error" in fork["results"][3]
        for item, a, b in zip(items[:3], ref["results"], fork["results"]):
            alone = _both(servers, "/v1/neuralese/decide", "POST",
                          {"messages": item["messages"], "options": item["continuations"]})
            for name, batched in (("reference", a), ("fork", b)):
                single = alone[name][1]
                assert batched["tokens"] == single["tokens"], (name, batched, single)
                assert all(abs(x - y) <= 1e-3 for x, y in zip(batched["log_probs"], single["log_probs"])), (name, batched, single)
            assert all(abs(x - y) <= ATOL_LOGPROB for x, y in zip(a["log_probs"], b["log_probs"])), (a, b)


def test_capabilities_the_fork_does_not_serve_fail_loudly(servers):
    status, body = _json(servers["fork"] + "/v1/neuralese/grad", "POST", {"terms": []})
    assert status == 501 and body["error"]["code"] == "neuralese-grad-unavailable"
    status, adapter = _json(servers["reference"] + "/v1/neuralese/adapters", "POST", {"kind": "xs", "rank": 2})
    assert status == 201
    # An adapter the fork has no LoRA for, and one decoded through a projection, fail; never the base model.
    bound = [{"id": adapter["id"], "scale": 1.0}]
    status, body = _json(servers["fork"] + "/v1/chat/completions", "POST",
                         {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 2, "x_natlang_adapters": bound})
    assert status == 409 and body["error"]["code"] == "neuralese-adapter-not-loaded"
    status, body = _json(servers["fork"] + "/v1/neuralese/decide", "POST",
                         {"messages": [{"role": "user", "content": "hi"}], "options": ["a", "b"],
                          "adapters": [{"code": adapter["id"], "projection": "p", "scale": 1.0}]})
    assert status == 501 and body["error"]["code"] == "neuralese-adapters-projection-unavailable"


REFERENCE_ONLY = {"adapters.create", "adapters.direct", "adapters.lora-export", "adapters.projection", "grad", "init-body",
                  "grad.order2", "optim"}


def _error_code(response):
    status, payload = response
    try:
        return status, json.loads(payload)["error"]["code"]
    except (ValueError, KeyError, TypeError):
        return status, f"<not the error envelope: {payload[:80]!r}>"


def test_one_capability_model_on_every_runtime(servers):
    """spec/NEURALESE_PORT.md "Capabilities": each runtime lists what it serves; what it lacks answers 501
    `neuralese-<capability>-unavailable` (never 404, never silently ignored); errors share one envelope."""
    ref = _json(servers["reference"] + "/v1/neuralese/info")[1]
    fork = _json(servers["fork"] + "/v1/neuralese/info")[1]
    assert ref["server"] == "reference" and fork["server"] == "llama.cpp"
    assert ref["capabilities"] == sorted(ref["capabilities"]) and fork["capabilities"] == sorted(fork["capabilities"])
    assert set(ref["capabilities"]) - set(fork["capabilities"]) == REFERENCE_ONLY
    # store.owners: the fork has owner-scoped holds; the reference lists it once its store implements them.
    assert set(fork["capabilities"]) - set(ref["capabilities"]) - {"store.owners"} == {"adapters.lora-load"}
    lacking = {  # capability -> a request needing it
        "grad": ("POST", "/v1/neuralese/grad", {"terms": []}),
        "optim": ("POST", "/v1/neuralese/optim", {"optimizer": "sgd", "params": [], "grads": []}),
        "adapters.create": ("POST", "/v1/neuralese/adapters", {"kind": "xs", "rank": 2}),
        "init-body": ("POST", "/v1/neuralese/init_body", {"messages": [], "placeholder": "<block>", "text": "x"}),
        "adapters.lora-export": ("GET", "/v1/neuralese/adapters/nz1_aaaa/lora", None),
    }
    for capability, (method, path, body) in lacking.items():
        assert capability not in fork["capabilities"]
        got = _error_code(_request(servers["fork"] + path, method, body))
        assert got == (501, "neuralese-" + capability.replace(".", "-") + "-unavailable"), (capability, got)
    got = _error_code(_request(servers["reference"] + "/v1/neuralese/adapters/nz1_aaaa/lora", "PUT", raw=b"x"))
    assert got == (501, "neuralese-adapters-lora-load-unavailable"), got
    for name in ("reference", "fork"):
        base = servers[name]
        for method in ("DELETE", "PATCH", "OPTIONS"):
            assert _error_code(_request(base + "/v1/neuralese/decide", method)) == (405, "method-not-allowed"), (name, method)
        assert _request(base + "/v1/neuralese/info", "HEAD")[0] == 405, name  # no body to read
        # A collect whose `referenced` is not a list of IDs is refused too.
        assert _error_code(_request(base + "/v1/neuralese/collect", "POST", {"referenced": "nz1_aaaa"})) == (400, "bad-json"), name
        assert _error_code(_request(base + "/v1/no-such-path", "POST", {})) == (404, "not-found"), name
        assert _error_code(_request(base + "/v1/neuralese/decide", "POST", [1, 2])) == (400, "bad-json"), name
        # A malformed collect is refused, not read as "nothing referenced" (which would drop unpinned blocks).
        assert _error_code(_request(base + "/v1/neuralese/collect", "POST", raw=b"{not json")) == (400, "bad-json"), name


def test_batched_scoring_shares_a_prefill_across_non_adjacent_items_in_request_order(servers):
    france = [{"role": "user", "content": "Is Paris the capital of France? Reply with a JSON value."}]
    water = [{"role": "user", "content": "Is water wet? Reply with a JSON value."}]
    items = [{"messages": france, "continuations": ["true", "false"]},
             {"messages": water, "continuations": ["true", "false"]},
             {"messages": france, "continuations": ["false", "true"]}]
    got = _both(servers, "/v1/natlang/score", "POST", {"items": items})
    for name in ("reference", "fork"):
        status, body = got[name]
        assert status == 200, (name, body)
        first, _, last = body["results"]
        assert first["tokens"] == last["tokens"][::-1], (name, body)
        assert all(abs(a - b) <= 1e-3 for a, b in zip(first["log_probs"], last["log_probs"][::-1])), (name, body)


def test_a_weight_adapter_served_as_a_lora_agrees(servers):
    """An xs adapter (coefficients on the reference server) exported as a GGUF LoRA for the fork: the same decision
    log-probabilities and the same greedy reply, both different from the base model's."""
    from natlang_neuralese.model.tiny_adapters import AdapterSpec
    from natlang_neuralese.serve.store import encode_block, make_block

    engine = servers["engine"]
    bank = engine.adapter_bank
    spec = bank.spec(kind="xs", rank=4, cutoff=engine.heads.cutoff)
    torch.manual_seed(11)
    coefficients = 1.5 * torch.randn(len(bank.matrices(spec)), spec.width)
    block = make_block(coefficients, spec.dialect(), type="Adapter")
    assert AdapterSpec.parse(block.dialect) == spec
    status, _ = _request(f"{servers['reference']}/v1/neuralese/blocks/{block.id}", "PUT", raw=encode_block(block))
    assert status == 201
    # The reference server exports its adapter as a GGUF LoRA (what a runtime fetches for the fork).
    status, gguf = _request(f"{servers['reference']}/v1/neuralese/adapters/{block.id}/lora")
    assert status == 200 and gguf[:4] == b"GGUF"
    status, loaded = _request(f"{servers['fork']}/v1/neuralese/adapters/{block.id}/lora", "PUT", raw=gguf)
    assert status == 201, loaded
    question = {"messages": [{"role": "user", "content": "Is Paris the capital of France? Reply with a JSON value."}],
                "options": ["true", "false"]}
    base = _both(servers, "/v1/neuralese/decide", "POST", question)
    adapted = _both(servers, "/v1/neuralese/decide", "POST", {**question, "adapters": [{"id": block.id, "scale": 1.0}]})
    for name in ("reference", "fork"):
        assert adapted[name][0] == 200, adapted[name]
    ref, fork = adapted["reference"][1]["log_probs"], adapted["fork"][1]["log_probs"]
    print(f"adapted decide: reference {ref} fork {fork}; base {base['reference'][1]['log_probs']}")
    assert max(abs(a - b) for a, b in zip(ref, base["reference"][1]["log_probs"])) > 10 * ATOL_LOGPROB
    assert max(abs(a - b) for a, b in zip(ref, fork)) <= ATOL_LOGPROB, (ref, fork)
    reply = {"messages": [{"role": "user", "content": "Name a city in France."}], "max_tokens": 8, "temperature": 0}
    got = _both(servers, "/v1/chat/completions", "POST", {**reply, "x_natlang_adapters": [{"id": block.id, "scale": 0.5}]})
    texts = {name: got[name][1]["choices"][0]["message"].get("content") for name in got}
    assert texts["reference"] == texts["fork"], texts
    # The binding is per request: the next request without adapters is the base model again.
    again = _both(servers, "/v1/neuralese/decide", "POST", question)
    assert again["fork"][1]["log_probs"] == base["fork"][1]["log_probs"]


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


def test_view_plans_agree(servers):
    """The fixture's write site (one template write), faithful (no instructions), and a long value at a small window
    (part writes and a combine write)."""
    from pathlib import Path

    fixture = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "view-site.json").read_text())
    # A view site's prompt renders identically on both servers and the stop logits agree within 2e-3, but the payload
    # at its first position already differs by up to 2.4e-2 (f32 CPU kernels, measured 2026-10-10 on both stop sources):
    # view writes are compared at the chunked plan's tolerance.
    atol = 3 * ATOL_PAYLOAD
    ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/view", "POST", fixture["site"]), atol=atol)
    assert ref["parts"] == fork["parts"] == 1
    # The default window: the served context less the site's prompt without the value, the reply and a margin. Both
    # servers serve CONTEXT, so the windows agree, for each site (instructions lengthen the site's prompt).
    assert ref["window"] == fork["window"] < CONTEXT - 6 - 64 - 32
    faithful = {"value": fixture["site"]["value"]}
    ref_faithful, fork_faithful = _block_agrees(servers, _both(servers, "/v1/neuralese/view", "POST", faithful), atol=atol)
    assert ref_faithful["parts"] == fork_faithful["parts"] == 1
    assert ref_faithful["window"] == fork_faithful["window"] > ref["window"]
    long = {**fixture["site"], "value": json.dumps({f"line{i}": f"fee {i} paid" for i in range(12)}), "window": 16}
    ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/view", "POST", long), atol=3 * ATOL_PAYLOAD)
    assert ref["parts"] == fork["parts"] > 1 and ref["window"] == fork["window"] == 16


def _small_context_pair(servers, context: int):
    """A second pair from the same weights serving `context` tokens: a reference Engine sharing the fixture's backbone
    and heads, and the fork's command with its -c replaced."""
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.http import serve
    from natlang_neuralese.serve.store import TensorStore

    base = servers["engine"]
    engine = Engine(base.backbone, base.heads, base.tokenizer, TensorStore(), base.dialect, max_block=base.max_block,
                    context=context)
    engine.start()
    reference = serve(engine)
    threading.Thread(target=reference.serve_forever, daemon=True).start()
    command = list(servers["command"])
    command[command.index("-c") + 1] = str(context)
    fork, fork_url = _start(command)
    host, port = reference.server_address[:2]
    return {"reference": f"http://{host}:{port}", "fork": fork_url, "process": fork, "server": reference, "engine": engine}


def test_view_default_window_chunks_the_same_at_the_served_context(servers):
    """With no `window`, view plans against the served context of the server writing it (spec: view): two servers
    configured with the same context report the same context and window and cut a value longer than one site into the
    same parts. The value is a little over one window, so a small context keeps the part writes short."""
    small = _small_context_pair(servers, 1024)
    try:
        infos = {name: _json(small[name] + "/v1/neuralese/info")[1] for name in ("reference", "fork")}
        assert infos["reference"]["context"] == infos["fork"]["context"] == 1024
        probe = _both(small, "/v1/neuralese/view", "POST", {"value": "short"})
        window = probe["reference"][1]["window"]
        assert probe["fork"][1]["window"] == window and 256 <= window < 1024
        tokenizer = servers["engine"].tokenizer
        value = " ".join(f"fee {i} paid on day {i % 28 + 1}." for i in range(400))
        ids = tokenizer(value, add_special_tokens=False)["input_ids"]
        value = tokenizer.decode(ids[:window + window // 2])
        n = len(tokenizer(value, add_special_tokens=False)["input_ids"])
        assert window < n <= 2 * window
        ref, fork = _block_agrees(small, _both(small, "/v1/neuralese/view", "POST", {"value": value}), atol=3 * ATOL_PAYLOAD)
        assert ref["parts"] == fork["parts"] == 2 and ref["window"] == fork["window"] == window
    finally:
        small["process"].terminate()
        small["server"].shutdown()
        small["engine"].stop()


def test_tool_call_arguments_are_the_same_canonical_text(servers):
    """`function.arguments` is compact JSON text (no spaces after separators), with non-ASCII characters as UTF-8 and
    keys in the order the model wrote them (spec "Response fields"); both servers answer the same bytes, streamed or
    not, for LFM2's Pythonic calls and the Qwen family's JSON calls."""
    plans = [
        ['<|tool_call_start|>[f(zeta="caf\u00e9 \u2713", alpha=[1, 2.5, True, None], nested={"b": 1, "a": "x\\ny \\"q\\""})]'
         '<|tool_call_end|>'],
        ['<tool_call>\n{"name": "g", "arguments": {"z": "\u00fc", "a": {"k": [1, -2, 0.5]}, "m": "tab\\tend"}}\n</tool_call>'],
    ]
    for plan in plans:
        body = {"messages": OPENING, "max_tokens": 128, "x_natlang_forced": plan}
        got = _both(servers, "/v1/chat/completions", "POST", body)
        texts = {}
        for name, (status, reply) in got.items():
            assert status == 200, (name, reply)
            calls = reply["choices"][0]["message"].get("tool_calls") or []
            assert calls, (name, reply["choices"][0]["message"])
            texts[name] = [c["function"]["arguments"] for c in calls]
            for text in texts[name]:
                assert text == json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False), (name, text)
            status, _, events = _stream(servers[name], body)
            streamed, _ = _assemble(events)
            assert [c["function"]["arguments"] for c in streamed["calls"]] == texts[name], name
        assert texts["reference"] == texts["fork"]
        assert all(any(ord(ch) > 127 for ch in text) for text in texts["reference"]), texts


def _stream_then_leave(url: str, body: dict) -> int:
    """Open a streamed chat completion, read until the first content delta, and close the connection: the deltas read."""
    import http.client
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=600)
    connection.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
                       {"content-type": "application/json"})
    response = connection.getresponse()
    assert response.status == 200
    seen = 0
    while seen < 2:
        line = response.fp.readline()
        assert line, "the stream ended before any content"
        if line.startswith(b"data: ") and b'"content"' in line:
            seen += 1
    response.close()
    connection.close()
    return seen


def test_a_client_that_leaves_a_stream_stops_generation(servers):
    """A streamed reply whose client disconnects stops generating (spec: streaming) instead of running out its
    max_tokens: the reference drops the sequence from its engine; the fork, which serves one request at a time, is free
    for the next request at once. The reply is a long forced text, so it would otherwise run for thousands of steps."""
    import time

    if servers["process"].args[0].endswith("node"):
        pytest.skip("the wasm module serves in process: Node cannot see the client leave while the module runs")
    words = " ".join(f"w{i}" for i in range(2500))
    body = {"messages": [{"role": "user", "content": "Repeat the words."}], "max_tokens": 4000, "x_natlang_forced": [words]}
    engine = servers["engine"]
    _stream_then_leave(servers["reference"], body)
    deadline = time.time() + 30
    while engine._active and time.time() < deadline:
        time.sleep(0.05)
    assert not engine._active, "the reference kept generating for a client that left"
    # The fork: a full forced reply of these words takes far longer than a short request after the client left.
    start = time.time()
    status, _ = _json(servers["fork"] + "/v1/chat/completions", "POST", {**body, "max_tokens": 200})
    assert status == 200
    per_token = (time.time() - start) / 200
    _stream_then_leave(servers["fork"], body)
    start = time.time()
    status, _ = _json(servers["fork"] + "/v1/chat/completions", "POST",
                      {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})
    assert status == 200
    waited = time.time() - start
    assert waited < 0.25 * per_token * 2500, f"the fork kept generating for a client that left ({waited:.1f} s)"


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
        ("<|tool_call_start|>[eval(code='const a = 1;\\n" + "positiveLines = positiveLines.filter(f);\\n" * 4, {"repeat": 3}),
        ("<|tool_call_start|>[evaluate(code='1')]", {"tools": ["eval"]}),
        ("<|tool_call_start|>[eval(code='const t = `a ${b}`;\\nconst o = {a: 1, b: [2, 3]};\\nawait f(o)\\n')]", {}),
        ("<|tool_call_start|>[eval(code='const s = 1;\\nconst t = " + "l.line_amount||" * 5, {}),
        ("<|tool_call_start|>[eval(code='const p = a.filter(f);\\nconst p = p.filter(g);\\nreturn p;')]", {}),
    ]
    replies.extend([
        ("<|tool_call_start|>[eval(code='return [Math.max(1, 2)];')]<|tool_call_end|>", {"tools": ["eval"]}),
        ("<|tool_call_start|>[eval(code='return [Math.max(1, 2)];'), forbidden(value=1)]", {"tools": ["eval"]}),
    ])
    verdicts = []
    for reply, guidance in replies:
        got = _both(servers, "/v1/neuralese/guidance/check", "POST", {"reply": reply, "guidance": guidance})
        assert got["reference"][1] == got["fork"][1], (reply, got)
        verdicts.append(got["reference"][1]["reason"])
    assert verdicts == [None, "syntax", "repetition", "unknown-tool", None, "repetition", "redeclaration", None, "unknown-tool"]
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
                     {"role": "tool", "tool_call_id": "c0", "content": "1"}], "tools": tools},
                 # A reply that returned a written value, as replies carry it: a part list inside the arguments JSON.
                 {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [
                     {"id": "c1", "type": "function", "function": {"name": "return_result", "arguments": json.dumps(
                         {"status": "success", "value": [{"type": "neuralese", "id": "nz1_" + "a" * 52}]})}}]},
                     {"role": "user", "content": "next"}]}):
        got = _both(servers, "/v1/neuralese/render", "POST", body)
        assert got["reference"][1]["prompt"] == got["fork"][1]["prompt"], (got["reference"][1]["prompt"], got["fork"][1]["prompt"])
        if "c1" in json.dumps(body):
            assert "nz1_" not in got["reference"][1]["prompt"], "the written value renders as a block, not as JSON text"


# Parity of the fork with the reference (2026-10-10): info fields, embed, template value types and argument paths,
# parts typed "unknown", Qwen-style calls and thinking, streaming, shared error behaviour.

def test_info_fields_agree_in_shape(servers):
    """Both report the same fields with the same types; values say what each serves."""
    ref = _json(servers["reference"] + "/v1/neuralese/info")[1]
    fork = _json(servers["fork"] + "/v1/neuralese/info")[1]
    assert set(ref) == set(fork), (sorted(ref), sorted(fork))
    assert {k: type(v) for k, v in ref.items()} == {k: type(v) for k, v in fork.items()}
    assert ref["stream"] is fork["stream"] is True and "chat.stream" in fork["capabilities"]
    assert ref["grad"] is True and fork["grad"] is False and ref["grad_order"] == 2 and fork["grad_order"] == 0
    assert fork["adapters"] == [] and fork["projections"] == {} and fork["store"] == {"owners": True, "persistent": False}


def test_embed_agrees(servers):
    body = {"text": "count the fees", "type": "Neuralese<string>"}
    ref, fork = _block_agrees(servers, _both(servers, "/v1/neuralese/embed", "POST", body), atol=1e-6)
    assert ref["id"] == fork["id"], "token embeddings are exact: the same content ID"
    assert ref["type"] == fork["type"] == "Neuralese<string>"
    assert ref["producer"] == fork["producer"] == {"kind": "text-init", "text": "count the fees"}
    got = _both(servers, "/v1/neuralese/embed", "POST", {"text": ""})
    assert _codes(got) == {(400, "neuralese-embed")}


def _codes(got):
    return {(status, body["error"]["code"]) for status, body in got.values()}


def _normalized(message, blocks):
    """A reply message with block IDs replaced by their index among `blocks` and call IDs dropped, for comparing the
    two servers (their payloads differ in the last bits, so their content IDs differ)."""
    index = {meta["id"]: n for n, meta in enumerate(blocks)}

    def walk(value):
        if isinstance(value, dict):
            return {k: (f"#{index.get(v, v)}" if k == "id" and isinstance(v, str) and v.startswith("nz1_") else walk(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    message = json.loads(json.dumps(message))
    for call in message.get("tool_calls") or []:
        call.pop("id", None)
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return walk(message)


def _replies_agree(servers, body):
    got = _both(servers, "/v1/chat/completions", "POST", body)
    (rs, ref), (fs, fork) = got["reference"], got["fork"]
    assert rs == fs == 200, (ref, fork)
    messages = {name: _normalized(got[name][1]["choices"][0]["message"], got[name][1]["neuralese"]["blocks"]) for name in got}
    assert messages["reference"] == messages["fork"], messages
    rb, fb = ref["neuralese"]["blocks"], fork["neuralese"]["blocks"]
    assert [b["length"] for b in rb] == [b["length"] for b in fb] and [b.get("type") for b in rb] == [b.get("type") for b in fb]
    for a, b in zip(rb, fb):
        diff = (_payload(servers["reference"], a["id"]) - _payload(servers["fork"], b["id"])).abs().max().item()
        assert diff <= ATOL_PAYLOAD, f"payload differs by {diff:.3g}"
    assert ref["choices"][0]["finish_reason"] == fork["choices"][0]["finish_reason"]
    assert ref["usage"] == fork["usage"]
    return ref, fork, messages["reference"]


OPENING = [{"role": "user", "content": "Combine the two notes into one value."}]


def test_template_value_type_unknown_agrees(servers):
    """value_type "unknown": the value written unquoted (native value syntax), the block typed Neuralese<unknown>, its
    part marked value_type "unknown"."""
    template = {"call": "return_result", "arguments": {"status": "success"}, "value": "write", "value_type": "unknown"}
    ref, fork, message = _replies_agree(servers, {"messages": OPENING, "max_tokens": 64, "neuralese_template": template})
    assert ref["neuralese"]["blocks"][0]["type"] == "Neuralese<unknown>"
    assert message["tool_calls"][0]["function"]["arguments"]["value"] == [
        {"type": "neuralese", "id": "#0", "value_type": "unknown"}], message
    got = _both(servers, "/v1/chat/completions", "POST", {"messages": OPENING, "max_tokens": 8, "neuralese_template": {
        **template, "value_type": "number"}})
    assert _codes(got) == {(400, "neuralese-template")}


def test_template_argument_path_agrees(servers):
    """argument_path: the value written (or decoded) inside nested arguments; a path that does not start at the
    argument is refused."""
    arguments = {"status": "success", "value": {"note": "x", "count": 2}}
    template = {"call": "return_result", "arguments": arguments, "argument_path": ["value", "note"], "value": "write"}
    _, _, message = _replies_agree(servers, {"messages": OPENING, "max_tokens": 64, "neuralese_template": template})
    assert message["tool_calls"][0]["function"]["arguments"]["value"]["note"] == [{"type": "neuralese", "id": "#0"}], message
    _replies_agree(servers, {"messages": OPENING, "max_tokens": 64, "neuralese_template": {**template, "value_type": "unknown"}})
    _replies_agree(servers, {"messages": OPENING, "max_tokens": 12, "neuralese_template": {**template, "value": "decode"}})
    for path in (["note"], ["value", "missing"], "value"):
        got = _both(servers, "/v1/chat/completions", "POST", {"messages": OPENING, "max_tokens": 8, "neuralese_template": {
            **template, "argument_path": path}})
        assert _codes(got) == {(400, "neuralese-template")}, (path, got)


def test_parts_typed_unknown_render_unquoted(servers):
    """A tool-call argument value that is exactly a block with value_type "unknown" (given, or the stored block's type
    Neuralese<unknown>) renders unquoted; a "string" one stays quoted."""
    from natlang_neuralese.serve.store import encode_block, make_block

    width = _both(servers, "/v1/neuralese/info")["reference"][1]["width"]
    torch.manual_seed(13)
    typed = make_block(0.05 * torch.randn(2, width), heads_dialect(), type="Neuralese<unknown>")
    for name in ("reference", "fork"):
        assert _request(f"{servers[name]}/v1/neuralese/blocks/{typed.id}", "PUT", raw=encode_block(typed))[0] == 201
    plain = "nz1_" + "b" * 52

    def turn(part):
        return {"messages": [{"role": "user", "content": "go"}, {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "return_result", "arguments": json.dumps(
                {"status": "success", "value": [part]})}}]}, {"role": "user", "content": "next"}]}

    prompts = {}
    for label, part in (("given", {"type": "neuralese", "id": plain, "value_type": "unknown"}),
                        ("stored", {"type": "neuralese", "id": typed.id}),
                        ("string", {"type": "neuralese", "id": typed.id, "value_type": "string"})):
        got = _both(servers, "/v1/neuralese/render", "POST", turn(part))
        assert got["reference"][0] == got["fork"][0] == 201, got
        assert got["reference"][1]["prompt"] == got["fork"][1]["prompt"], got
        prompts[label] = got["reference"][1]["prompt"]
    assert "value=<block>" in prompts["given"] and prompts["given"] == prompts["stored"], prompts
    assert "value=<block>" not in prompts["string"], prompts
    # A reply carrying the typed part is read the same (greedy continuation).
    body = {**turn({"type": "neuralese", "id": typed.id, "value_type": "unknown"}), "max_tokens": 6}
    _replies_agree(servers, body)


def test_qwen_calls_and_lone_think_close_parse_the_same(servers):
    """Model output in the Qwen family's forms (Maple, Mellum): `<tool_call>{json}</tool_call>` and a reply that only
    closes `</think>` (thinking templates open it in the generation prompt); markup that does not parse stays text."""
    write = {"neuralese": "write"}
    plans = [
        ["Weighing the notes.</think>Answer: ", write, " done"],
        ['<tool_call>\n{"name": "return_result", "arguments": {"status": "success", "value": "', write, '"}}\n</tool_call>'],
        ["<think>short</think>", '<|tool_call_start|>[eval(code="1")]<|tool_call_end|><tool_call>{"name": "f", '
         '"arguments": "{\\"a\\": [1, true, null]}"}</tool_call>'],
        ["Before <tool_call>not json</tool_call> after"],
    ]
    expected = []
    for plan in plans:
        _, _, message = _replies_agree(servers, {"messages": OPENING, "max_tokens": 64, "x_natlang_forced": plan})
        expected.append(message)
    assert expected[0]["reasoning_content"] == "Weighing the notes." and expected[0]["content"][1] == {"type": "neuralese", "id": "#0"}
    assert expected[1]["tool_calls"][0]["function"] == {"name": "return_result", "arguments": {
        "status": "success", "value": [{"type": "neuralese", "id": "#0"}]}} and expected[1]["content"] is None
    assert [c["function"]["name"] for c in expected[2]["tool_calls"]] == ["eval", "f"]
    assert expected[2]["tool_calls"][1]["function"]["arguments"] == {"a": [1, True, None]}
    assert expected[3]["content"] == "Before <tool_call>not json</tool_call> after" and "tool_calls" not in expected[3]


def _stream(url: str, body: dict) -> tuple[int, str, list]:
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps({**body, "stream": True}).encode(),
                                     method="POST", headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        kind = response.headers.get("content-type", "")
        text = response.read().decode()
    events = [line[len("data: "):] for line in text.split("\n") if line.startswith("data: ")]
    return response.status, kind, events


def _assemble(events: list) -> tuple[dict, dict]:
    """A client's view of a stream (ts-host chat-completion.ts assembleChatCompletion): the deltas folded in order, and
    the final chunk."""
    assert events[-1] == "[DONE]", events[-3:]
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    parts, calls, blocks = [], [], []
    for chunk in chunks[1:-1]:
        delta = chunk["choices"][0]["delta"]
        if isinstance(delta.get("content"), str):
            if parts and parts[-1]["type"] == "text":
                parts[-1]["text"] += delta["content"]
            else:
                parts.append({"type": "text", "text": delta["content"]})
        elif isinstance(delta.get("content"), list):
            assert delta["content"][0]["id"] == chunk["neuralese"]["block"]["id"]
            parts.extend(delta["content"])
            blocks.append(chunk["neuralese"]["block"])
        calls.extend(delta.get("tool_calls") or [])
    final = chunks[-1]
    assert final["choices"][0]["delta"] == {} and final["choices"][0]["finish_reason"]
    return {"parts": parts, "calls": calls, "blocks": blocks}, final


STREAMED = [
    {"messages": [{"role": "user", "content": "Name a city in France."}], "max_tokens": 10},
    {"messages": OPENING, "max_tokens": 64, "x_natlang_forced": ["Plan: ", {"neuralese": "write"}, " done"]},
    {"messages": OPENING, "max_tokens": 64, "neuralese_template": {"call": "return_result", "arguments": {
        "status": "success"}, "value": "write"}},
    {"messages": OPENING, "max_tokens": 64, "x_natlang_forced": [
        'Sure. <tool_call>{"name": "f", "arguments": {"v": "', {"neuralese": "write"}, '"}}</tool_call>']},
]


def test_streamed_reply_assembled_equals_the_non_streamed_reply(servers):
    """`"stream": true` on both servers: the role delta, text deltas held back at call markup, a part per written block
    (with its meta), the parsed calls as one tool_calls delta, and a final chunk whose x_natlang_message is the message
    a non-streaming request returns; the two servers stream the same."""
    seen = {}
    for body in STREAMED:
        for name in ("reference", "fork"):
            status, kind, events = _stream(servers[name], body)
            assert status == 200 and kind.startswith("text/event-stream"), (name, kind)
            streamed, final = _assemble(events)
            whole = _json(servers[name] + "/v1/chat/completions", "POST", body)[1]
            message = whole["choices"][0]["message"]
            blocks = final["neuralese"]["blocks"]
            # blocks written inside a call are part of the call's arguments, not content deltas
            calls = final["x_natlang_message"].get("tool_calls")
            assert streamed["blocks"] == (blocks[:len(streamed["blocks"])] if calls else blocks), name
            assert _normalized(final["x_natlang_message"], blocks) == _normalized(message, whole["neuralese"]["blocks"]), name
            assert final["usage"] == whole["usage"] and final["choices"][0]["finish_reason"] == whole["choices"][0]["finish_reason"]
            assert [(c["index"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in streamed["calls"]] == \
                [(i, c["function"]["name"], json.loads(c["function"]["arguments"]))
                 for i, c in enumerate(final["x_natlang_message"].get("tool_calls") or [])], name
            text = "".join(p["text"] for p in streamed["parts"] if p["type"] == "text")
            assert "<|tool_call_start|>" not in text and "<tool_call>" not in text, (name, text)
            content = final["x_natlang_message"]["content"]
            if not final["x_natlang_message"].get("tool_calls"):
                # without calls the deltas are the content (up to surrounding whitespace)
                if isinstance(content, str):
                    assert text.strip() == content, (name, text, content)
                elif content:
                    assert [p["type"] for p in streamed["parts"]] == [p["type"] for p in content], name
            seen[name] = (_normalized(final["x_natlang_message"], blocks), text)
        assert seen["reference"] == seen["fork"], seen


def test_streaming_reports_errors_as_events_and_guidance(servers):
    """A stream that fails after it started ends with an error event and [DONE]; guided streams carry
    x_natlang_guidance in the final chunk; `guidance: {}` is guidance with every default on both."""
    unknown = {"messages": [{"role": "user", "content": [{"type": "neuralese", "id": "nz1_" + "c" * 52}]}], "max_tokens": 4}
    for name in ("reference", "fork"):
        status, _, events = _stream(servers[name], unknown)
        assert status == 200 and events[-1] == "[DONE]", (name, events)
        assert json.loads(events[-2])["error"]["code"] == "neuralese-unknown-block", (name, events)
    tools = [{"type": "function", "function": {"name": "return_result", "parameters": {}}}]
    body = {"messages": [{"role": "user", "content": "Count the fees."}], "max_tokens": 16, "tools": tools,
            "tool_choice": "required", "guidance": {}}
    finals = {}
    for name in ("reference", "fork"):
        whole = _json(servers[name] + "/v1/chat/completions", "POST", body)[1]
        assert "x_natlang_guidance" in whole, (name, whole)
        _, final = _assemble(_stream(servers[name], body)[2])
        assert final["x_natlang_guidance"] == whole["x_natlang_guidance"], name
        finals[name] = final["x_natlang_guidance"]
    assert finals["reference"] == finals["fork"]


def test_batched_item_errors_and_block_width_agree(servers):
    """Batched scoring reports a failing item's error as "code: detail" on both; reading a block of another width is
    neuralese-bad-block on both."""
    from natlang_neuralese.serve.store import encode_block, make_block

    items = [{"messages": OPENING, "continuations": []}, "not an item"]
    got = _both(servers, "/v1/neuralese/decide_many", "POST", {"items": items})
    assert got["reference"][1] == got["fork"][1], got
    assert got["reference"][1]["results"][0]["error"].startswith("neuralese-decision: ")
    width = _both(servers, "/v1/neuralese/info")["reference"][1]["width"]
    narrow = make_block(torch.zeros(2, width // 2), heads_dialect())
    for name in ("reference", "fork"):
        assert _request(f"{servers[name]}/v1/neuralese/blocks/{narrow.id}", "PUT", raw=encode_block(narrow))[0] == 201
    body = {"messages": [{"role": "user", "content": [{"type": "neuralese", "id": narrow.id}]}], "max_tokens": 2}
    assert _codes(_both(servers, "/v1/chat/completions", "POST", body)) == {(400, "neuralese-bad-block")}


def test_max_tokens_zero_generates_nothing_and_bad_limits_fail(servers):
    """SERVER_HARMONIZATION #20: `max_tokens` (else `max_completion_tokens`) is honoured as given, 0 included: no generated
    token, the same empty message and finish_reason "length" on both, streamed and not (a template readout too); absent
    or null is 512; a negative, fractional or non-numeric limit is 400 bad-max-tokens on both, before a stream opens."""
    for body in ({"messages": OPENING, "max_tokens": 0},
                 {"messages": OPENING, "max_tokens": None, "max_completion_tokens": 0},
                 {"messages": OPENING, "max_tokens": 0, "neuralese_template": {"call": "return_result", "arguments": {
                     "status": "success"}, "value": "write"}}):
        ref, fork, message = _replies_agree(servers, body)
        assert ref["choices"][0]["finish_reason"] == "length" and ref["usage"]["completion_tokens"] == 0, ref
        assert not message.get("content") and not message.get("tool_calls") and not ref["neuralese"]["blocks"], message
        for name in ("reference", "fork"):
            _, final = _assemble(_stream(servers[name], body)[2])
            assert final["choices"][0]["finish_reason"] == "length" and final["usage"] == ref["usage"], (name, final)
    for bad in ({"max_tokens": -1}, {"max_tokens": 1.5}, {"max_tokens": "8"}, {"max_tokens": True},
                {"max_completion_tokens": -2}):
        for stream in (False, True):
            body = {"messages": OPENING, **bad, **({"stream": True} if stream else {})}
            got = {name: _request(servers[name] + "/v1/chat/completions", "POST", body) for name in ("reference", "fork")}
            codes = {(status, json.loads(payload)["error"]["code"]) for status, payload in got.values()}
            assert codes == {(400, "bad-max-tokens")}, (bad, stream, got)
