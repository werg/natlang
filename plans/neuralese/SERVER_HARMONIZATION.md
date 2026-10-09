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

## Endpoint × runtime matrix (after this change)

| Endpoint / feature | Reference | Fork (native, wasm) | Capability |
| --- | --- | --- | --- |
| `GET /health`, `/v1/health`, `/v1/models` | yes | yes | — |
| `GET /v1/neuralese/info` | yes, `server: "reference"`, `capabilities` | yes, `server: "llama.cpp"`, `capabilities` | — |
| `POST /v1/chat/completions` (forced plans, length hints, passes, template, guidance, adapters) | yes | yes | `chat`, `template` |
| … `stream: true` | yes (SSE) | 501 | `chat.stream` |
| … `neuralese_template.value_type: "unknown"` | yes | 501 | `template.value-type` |
| … `neuralese_template.argument_path` | yes | 501 | `template.argument-path` |
| … parts with `value_type: "unknown"` | yes | 501 (every endpoint that renders messages) | `parts.value-type` |
| … `x_natlang_adapters` by id | applied directly | needs a loaded LoRA, else 409 | `adapters.direct` / `adapters.lora-load` |
| … adapters as `{code, projection}` | yes | 501 | `adapters.projection` |
| `POST /v1/neuralese/decide` | yes | yes | `decide` |
| `POST /v1/natlang/score`, `/v1/neuralese/decide_many` | yes, grouped by (adapters, messages, tools) | yes, the same grouping | `score` |
| `POST /v1/neuralese/encode`, `write`, `view` (was `digest`; each view write a template write of view's body, DECISIONS.md 2026-10-09) | yes | yes | `encode`, `write`, `view` |
| `POST /v1/neuralese/render`, `guidance/check` | yes | yes | `render`, `guidance.check` |
| Block store: `PUT/GET /blocks/{id}`, `/meta`, `/pin`, `/unpin`, `POST /collect`, `x-natlang-owner` | yes | yes | `store`, `store.owners` |
| `POST /v1/neuralese/grad` | yes (order 1 and 2) | 501 | `grad`, `grad.order2` |
| `POST /v1/neuralese/optim` | yes | 501 | `optim` |
| `POST /v1/neuralese/embed` | yes | 501 | `embed` |
| `POST /v1/neuralese/adapters` | yes | 501 | `adapters.create` |
| `GET /v1/neuralese/adapters/{id}/lora` | yes | 501 | `adapters.lora-export` |
| `PUT /v1/neuralese/adapters/{id}/lora` | 501 | yes | `adapters.lora-load` |
| Unknown path | 404 `not-found` | 404 `not-found` | — |
| Unsupported method | 405 `method-not-allowed` | 405 `method-not-allowed` | — |
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
| 3 | `info.adapters` is a list on the reference and the string `"lora"` on the fork. | both info handlers | kept for older clients; `capabilities` is authoritative |
| 4–7 | The fork answered `embed`, `optim`, `POST adapters` and `GET …/lora` with 404 `not-found` (and `grad` with 501). | fork `handle` | fixed: 501 `neuralese-<capability>-unavailable` |
| 8 | The reference answered `PUT …/lora` with 404. | `http.py` `do_PUT` | fixed: 501 `neuralese-adapters-lora-load-unavailable` |
| 9 | Fork streaming: 400 `stream-unsupported`. | fork `handle` | fixed: 501 `neuralese-chat-stream-unavailable` |
| 10–12 | The fork silently ignored `neuralese_template.value_type`, `argument_path`, and parts with `value_type: "unknown"` (rendered them as strings). | fork template parsing, `render` | fixed: 501 with the capability's code |
| 13 | Projection adapters on the fork failed as `neuralese-adapters-unavailable`. | fork `bind_adapters` | fixed: renamed to `neuralese-adapters-projection-unavailable` |
| 14 | Batched scoring shared a prefill between consecutive same-prompt items only on the fork; the reference grouped across the request. | fork `decide_many` | fixed: the fork groups by (adapters, messages, tools) and answers in request order |
| 15 | Unsupported methods on the reference got Python's HTML 501 page. | `BaseHTTPRequestHandler` default | fixed: 405 JSON |
| 16 | A POST body that is valid JSON but not an object crashed the reference handler (no response). | `body.get` on a list | fixed: 400 `bad-json` |
| 17 | Any exception other than `RequestError`/`KeyError`/`JSONDecodeError` dropped the reference's connection. | `do_POST` except clauses | fixed: 500 `internal` envelope |
| 18 | **Fork bug:** `POST /collect` with a malformed body read "nothing referenced" and dropped every unpinned block. | fork `handle`: collect ran before the bad-json check | fixed: 400 `bad-json` first |
| 19 | The TS decide driver treated only 404 as "unsupported". | `neuralese-server.ts` | fixed: 404 or 501 |
| 20 | `max_tokens: 0`: the reference falls back to 512 (`or`), the fork honours 0. | `http.py` `_chat`; fork `generate` | open (edge case; would need an engine check for zero-length requests) |
| 21 | Per-item `error` strings in batched scoring differ. | both | accepted (documented in the spec) |
| 22 | Tool-call parsing: the fork parses Pythonic calls and a complete `<think>…</think>`; the reference also parses `<tool_call>` JSON and a lone `</think>`. | spec "Response fields" | accepted for now (spec row); port to the fork if a backbone emits JSON calls |
| 23 | Payload noise generators differ at `τ > 0`. | spec | inherent (different RNGs); documented |
| 24 | Concurrency: the reference serializes grad/decide/score/optim/encode under one lock and batches chat, write and view through the engine; the fork serializes everything under one mutex. | `http.py` `grad_lock`; fork `mutex` | same results, different throughput; documented |
| 25 | Owner-scoped store: the fork has it (915afccb7); the reference's Python side is uncommitted work in another session. | `git status` in the shared checkout | in progress elsewhere; the reference lists `store.owners` automatically once its store implements holds |

## Conformance

`tests/neuralese/test_server_conformance.py` has these new checks:

- `test_one_capability_model_on_every_runtime`: both servers report `server` and sorted
  `capabilities`, and the difference is exactly the reference-only set plus `adapters.lora-load`.
  Every lacking capability answers 501 with its code. Both servers answer the same 405, 404 and
  400 cases, including the malformed collect.
- `test_batched_scoring_shares_a_prefill_across_non_adjacent_items_in_request_order`

Both run for the native parameters (shallow, final) and the wasm parameter (final-wasm).
