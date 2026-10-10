# Neuralese server harmonization (2026-10-09)

Owner: "Can we harmonize our server behaviors?" This is the audit of the runtimes that serve
the Neuralese wire protocol, the divergences found, and the single capability model they now
follow. The normative contract is `spec/NEURALESE_PORT.md` ("Wire protocol", "Capabilities").

Runtimes:

- **Reference**: `training/neuralese/natlang_neuralese/serve/` (`http.py` routes, `grad.py`
  readouts and learning, `engine.py` generation).
- **Fork (native)**: `tools/neuralese/neuralese-service.cpp` in `werg/llama.cpp-neuralese`,
  pinned in `training/neuralese/llama-cpp-fork.json`.
- **Fork (wasm)**: the same C++ compiled to WebAssembly (`ts-host/vendor/neuralese-wasm`);
  `ts-host/scripts/neuralese-wasm-server.mjs` only forwards `(method, path, body)` to
  `service.handle`. Its API is the native one by construction, and
  `tests/neuralese/test_runtime_versions.py` pins its build to the fork commit.
- **vLLM path**: `serve/vllm_rollout.py` (`VllmNeuralese.generate`) is a Python library for
  rollouts, not an HTTP server. Its responses have the reference's chat shape; it has no info
  endpoint and needs no capability list. Adapters go through vLLM's LoRA path, and writes go
  through the reference port.
- **TS client**: `ts-host/src/model/neuralese-server.ts`, `neuralese-info.ts`, `scoring.ts`.

## Endpoint × runtime matrix (after fork parity, 2026-10-10)

| Endpoint / feature | Reference | Fork (native, wasm) | Capability |
| --- | --- | --- | --- |
| `GET /health`, `/v1/health`, `/v1/models` | yes | yes | — |
| `GET /v1/neuralese/info` | yes, `server: "reference"`, `capabilities` | yes, `server: "llama.cpp"`, `capabilities` | — |
| `POST /v1/chat/completions` (forced plans, length hints, passes, template, guidance, adapters) | yes | yes | `chat`, `template` |
| … `stream: true` | yes (SSE) | yes (SSE; wasm: event hook, in-process endpoint streams) | `chat.stream` |
| … `neuralese_template.value_type: "unknown"` | yes | yes | `template.value-type` |
| … `neuralese_template.argument_path` | yes | yes | `template.argument-path` |
| … parts with `value_type: "unknown"` | yes | yes | `parts.value-type` |
| … reply parsing: Pythonic and `<tool_call>` JSON calls, `<think>…</think>` and a lone `</think>` | yes | yes | — |
| … `x_natlang_adapters` by id | applied directly | needs a loaded LoRA, else 409 | `adapters.direct` / `adapters.lora-load` |
| … adapters as `{code, projection}` | yes | 501 | `adapters.projection` |
| `POST /v1/neuralese/decide` | yes | yes | `decide` |
| `POST /v1/natlang/score`, `/v1/neuralese/decide_many` | yes, grouped by (adapters, messages, tools) | yes, the same grouping | `score` |
| `POST /v1/neuralese/encode`, `write`, `view` (was `digest`; each view write a template write of view's body, DECISIONS.md 2026-10-09) | yes | yes | `encode`, `write`, `view` |
| `POST /v1/neuralese/render`, `guidance/check` | yes | yes | `render`, `guidance.check` |
| Block store: `PUT/GET /blocks/{id}`, `/meta`, `/pin`, `/unpin`, `POST /collect`, `x-natlang-owner` | yes | yes | `store`, `store.owners` |
| `POST /v1/neuralese/grad` | yes (order 1 and 2) | 501 | `grad`, `grad.order2` |
| `POST /v1/neuralese/optim` | yes | 501 | `optim` |
| `POST /v1/neuralese/embed` | yes | yes | `embed` |
| `POST /v1/neuralese/adapters` | yes | 501 | `adapters.create` |
| `GET /v1/neuralese/adapters/{id}/lora` | yes | 501 | `adapters.lora-export` |
| `PUT /v1/neuralese/adapters/{id}/lora` | 501 | yes | `adapters.lora-load` |
| Unknown path | 404 `not-found` | 404 `not-found` | — |
| Unsupported method (anything but GET, PUT, POST; HEAD included) | 405 `method-not-allowed` | 405 `method-not-allowed` | — |
| POST body not a JSON object | 400 `bad-json` | 400 `bad-json` (before any action) | — |
| Unexpected failure | 500 `internal` (envelope) | 500 `internal` (envelope) | — |

Defaults agree: `max_tokens` (or `max_completion_tokens`) 512, `temperature` 0,
`neuralese_temperature` 0, `max_block_length` capped by the server's hard maximum, BOS once at the
start of the rendered prompt, the shallow/final stop source fixed by the loaded heads. Errors are
`{"error": {"code", "message"}}` everywhere.

## Divergences found

| # | Divergence (before) | Evidence | Status |
| --- | --- | --- | --- |
| 1 | No capability list; clients inferred features from `adapters === "lora"` and probed for 404s. | `http.py` info; fork `handle` info; `neuralese-server.ts` `appliesLoras` | fixed: `capabilities` on both, `serves()` in the TS client |
| 2 | `server` reported by the fork only. | fork info | fixed: reference reports `"reference"` |
| 3 | `info.adapters` is a list on the reference and the string `"lora"` on the fork. | both info handlers | fixed (fork parity): the fork reports `adapters: []` (no kinds applied directly), `grad_order: 0`, `projections: {}`, `stream: true`: one field set, one type per field; `capabilities` stays authoritative |
| 4–7 | The fork answered `embed`, `optim`, `POST adapters` and `GET …/lora` with 404 `not-found` (and `grad` with 501). | fork `handle` | fixed: 501 `neuralese-<capability>-unavailable`; `embed` is now served by the fork |
| 8 | The reference answered `PUT …/lora` with 404. | `http.py` `do_PUT` | fixed: 501 `neuralese-adapters-lora-load-unavailable` |
| 9 | Fork streaming: 400 `stream-unsupported`. | fork `handle` | fixed: 501, then implemented (fork parity): the reference's chunk format, native through chunked transfer, wasm through `nzw_handle`'s event hook |
| 10–12 | The fork silently ignored `neuralese_template.value_type`, `argument_path`, and parts with `value_type: "unknown"` (rendered them as strings). | fork template parsing, `render` | fixed: 501, then implemented (fork parity) |
| 13 | Projection adapters on the fork failed as `neuralese-adapters-unavailable`. | fork `bind_adapters` | fixed: renamed to `neuralese-adapters-projection-unavailable` |
| 14 | Batched scoring shared a prefill between consecutive same-prompt items only on the fork; the reference grouped across the request. | fork `decide_many` | fixed: the fork groups by (adapters, messages, tools) and answers in request order |
| 15 | Unsupported methods on the reference got Python's HTML 501 page. | `BaseHTTPRequestHandler` default | fixed: 405 JSON (any `do_<METHOD>`); the fork answers HEAD 405 too instead of httplib's GET route |
| 16 | A POST body that is valid JSON but not an object crashed the reference handler (no response). | `body.get` on a list | fixed: 400 `bad-json` |
| 17 | Any exception other than `RequestError`/`KeyError`/`JSONDecodeError` dropped the reference's connection. | `do_POST` except clauses | fixed: 500 `internal` envelope |
| 18 | **Fork bug:** `POST /collect` with a malformed body read "nothing referenced" and dropped every unpinned block. | fork `handle`: collect ran before the bad-json check | fixed: 400 `bad-json` first |
| 19 | The TS decide driver treated only 404 as "unsupported". | `neuralese-server.ts` | fixed: 404 or 501 |
| 20 | `max_tokens: 0`: the reference falls back to 512 (`or`), the fork honours 0. | `http.py` `_chat`; fork `generate` | open (edge case; would need an engine check for zero-length requests) |
| 21 | Per-item `error` strings in batched scoring differ. | both | fixed (fork parity): `"<code>: <detail>"` for a request error on both, `"internal: …"` otherwise |
| 22 | Tool-call parsing: the fork parses Pythonic calls and a complete `<think>…</think>`; the reference also parses `<tool_call>` JSON and a lone `</think>`. | spec "Response fields" | fixed (fork parity): the fork ports chat.build_message (needed for Mellum/Maple) |
| 23 | Payload noise generators differ at `τ > 0`. | spec | inherent (different RNGs); documented |
| 24 | Concurrency: the reference serializes grad/decide/score/optim/encode under one lock and batches chat, write and view through the engine; the fork serializes everything under one mutex. | `http.py` `grad_lock`; fork `mutex` | same results, different throughput; documented |
| 25 | Owner-scoped store: the fork has it (915afccb7); the reference's Python side is uncommitted work in another session. | `git status` in the shared checkout | in progress elsewhere; the reference lists `store.owners` automatically once its store implements holds |

## Fork parity (2026-10-10)

The fork now serves every capability but `grad`, `grad.order2`, `optim`, `adapters.create`, `adapters.direct`,
`adapters.lora-export` and `adapters.projection` (learning and direct adapter application stay in the reference; the
fork applies adapters as loaded LoRAs). Further divergences closed on both sides:

| # | Divergence (before) | Status |
| --- | --- | --- |
| 26 | `guidance: {}` was off on the reference (Python truthiness) and on with defaults on the fork. | fixed: on with every default on both (spec: an object turns guidance on); absent, `null` or `false` is off |
| 27 | The reference read a block of another width (in its dialect) without a check; the fork answered `neuralese-bad-block`. | fixed: `Engine.lookup` checks the width (same message on both) |
| 28 | `POST /render` on the reference ignored stored `Neuralese<unknown>` types (generation used them). | fixed: render passes the store's types, as prefill does |
| 29 | `collect` with `referenced` not a list of IDs: reference read a string as characters, fork failed 500. | fixed: 400 `bad-json` on both |
| 30 | Streaming held back only LFM2's call marker; the reference's stream errors used code `server-error`. | fixed: `<|tool_call_start|>` and `<tool_call>` are held back on both; unexpected stream errors are `internal` |
| 32 | The threaded wasm build (`neuralese-wasm-mt`) under Node deadlocks intermittently (main thread in a futex wait, one ggml worker in another, the rest idle) after a few requests; every request it completed agreed. | open: conformance parameter `final-wasm-mt` is opt-in (`NATLANG_CONFORMANCE_WASM_MT=1`); suspected ggml's per-graph threadpool create/join under Emscripten pthreads |
| 31 | `view`'s default window: the model's `max_position_embeddings` on the reference, the served context (`min(n_ctx_train, -c)`) on the fork. | open: a value between the two windows is viewed in one site by the reference and chunked by the fork; a request `window` makes them agree |

## Conformance

`tests/neuralese/test_server_conformance.py` has these new checks:

- `test_one_capability_model_on_every_runtime`: both servers report `server` and sorted
  `capabilities`, and the difference is exactly the reference-only set plus `adapters.lora-load`.
  Every lacking capability answers 501 with its code. Both servers answer the same 405, 404 and
  400 cases, including the malformed collect.
- `test_batched_scoring_shares_a_prefill_across_non_adjacent_items_in_request_order`

Both run for the native parameters (shallow, final) and the wasm parameter (final-wasm).

Fork parity added (2026-10-10, fork 2d6557813): `test_info_fields_agree_in_shape`, `test_embed_agrees`,
`test_template_value_type_unknown_agrees`, `test_template_argument_path_agrees`, `test_parts_typed_unknown_render_unquoted`,
`test_qwen_calls_and_lone_think_close_parse_the_same`, `test_streamed_reply_assembled_equals_the_non_streamed_reply`,
`test_streaming_reports_errors_as_events_and_guidance`, `test_batched_item_errors_and_block_width_agree`; 75 passed
(shallow, final, final-wasm). `scripts/verify_neuralese_wasm_provenance.py` checks the vendored wasm's sha256s and fork
pin (also run by `tests/neuralese/test_runtime_versions.py`).
