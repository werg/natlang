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

Defaults agree: `max_tokens` (or `max_completion_tokens`) 512 (0 generates nothing; see #20), `temperature` 0,
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
| 20 | `max_tokens: 0`: the reference falls back to 512 (`or`), the fork honours 0. | `http.py` `_chat`; fork `generate` | fixed (fork 76f7af592): one rule on both (`http.py` `completion_limit`, the fork's `engine::completion_limit`): `max_tokens`, else `max_completion_tokens`; absent or `null` 512; a non-negative integer as given, 0 included (no generated token: empty message, `finish_reason` `length`, `completion_tokens` 0; the engines already checked the allowance before the first token); anything else (negative, fractional, string, boolean) 400 `bad-max-tokens`, for a streamed request too, as a plain response before any event (the fork's `streams()` is false for it). `test_max_tokens_zero_generates_nothing_and_bad_limits_fail` |
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
| 32 | The threaded wasm build (`neuralese-wasm-mt`) under Node deadlocked intermittently (main thread in a futex wait, one ggml worker in another, the rest idle), after 0 to 12 requests. | fixed (fork 8b41aecff): stacks (Node under gdb, V8 perf map, frame-pointer walk) showed the main thread in `ggml_threadpool_free` → `pthread_join` and the worker in emscripten's `_emscripten_thread_exit` waiting on its mailbox refcount (`_emscripten_thread_mailbox_shutdown`); ggml created and joined n_threads − 1 threads for every graph (every decoded token), and one of those exits never completed. The service now keeps one persistent ggml threadpool per thread count for the process's life, attached to the llama context (`llama_attach_threadpool`) and the CPU heads (`nz_heads_params.threadpool`): no thread exits while serving (natively too: no per-token thread churn). A replay loop that hung within 6 rounds ran 60 rounds (420 requests); `final-wasm-mt` runs by default again (repeated full runs: see Conformance) |
| 31 | `view`'s default window: the model's `max_position_embeddings` on the reference, the served context (`min(n_ctx_train, -c)`) on the fork; both subtracted a fixed 1024-token margin. | fixed: the window is what the writer attends to at the site, the serving server's served context (`info.context`; the reference gains `-c/--ctx-size` and `Engine(context=…)`, default `max_position_embeddings`) less the site's prompt rendered without the value (`Engine.prompt_positions` / the fork's `prompt_positions`), the reply (`max_block` + 64) and 32 tokens; `test_view_default_window_chunks_the_same_at_the_served_context` (both at `-c 1024`: same context, window and 2 parts) |
| 33 | A client leaving a streamed reply: the reference's handler died on the broken pipe while its engine generated on to `max_tokens`; the fork ignored the failed writes ("the request still completes") and held its one-request mutex. | fixed: both cancel. Reference: a failed write or a closed connection (polled every 0.25 s while there is nothing to send) sets `GenerationRequest.cancelled`, and the engine fails the sequence (`cancelled`) at its next round. Fork: `nz_event_sink` returns whether the event was delivered and `nz_service_handle` takes a `connected` probe (`sink.is_writable()`), asked before every generated position; `generate` stops with `cancelled`. The wasm hook cancels when `nzwOnEvent` returns `false`; under Node the module blocks the event loop, so its HTTP wrapper cannot see a client leave. `test_a_client_that_leaves_a_stream_stops_generation` (native pairs) |
| 34 | Tool-call `arguments` text: the reference `json.dumps` (`", "`/`": "` separators, non-ASCII as `\u` escapes), the fork compact UTF-8. | fixed: one canonical text, compact JSON, UTF-8, keys in the model's order (`chat.arguments_text`; the fork's `ordered_json::dump()`), streamed and not. The model emits Pythonic calls (LFM2) or JSON with its own spacing (Qwen), so there is no emitted text to copy; compact is the fork's existing form and what an OpenAI-style client receives from most servers; clients parse it. `test_tool_call_arguments_are_the_same_canonical_text` (byte equality) |
| 35 | `decide` after an earlier assistant turn with `reasoning_content` (the runtime's `scope_` turn): the fork required the full rendering (prompt + option) to extend the generation prompt and failed `neuralese-decision: the template renders the reply's prompt differently`, because LFM2.5's template (history policy last_turn_only) drops past reasoning once another assistant turn follows. The reference already scored the option after the prompt as generation renders it, cutting it from the full rendering after the reply's own generation prefix (`GradSession._target_items`). | fixed: the fork cuts the option the same way (after the last `<\|im_start\|>` prefix of the generation prompt, found in the full rendering; absent: `neuralese-decision`, as the reference now reports for decisions too). Earlier turns render as they do at generation time on both: the reference never scored a different prompt. The runtime's request shape was not the cause. `test_decision_after_an_assistant_turn_with_reasoning_scores_the_generation_prompt` (decide and score on both, and the score differs from the same turn without reasoning) |
| 36 | `POST /v1/neuralese/init_body` (reference-only capability `init-body`) answered 404 `not-found` on the fork, native and wasm. | fork `handle` | fixed (fork 1f4e334d5): 501 `neuralese-init-body-unavailable`, not listed in `info.capabilities`; the capability test now probes it; wasm re-vendored |

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

Follow-ups (2026-10-10, fork 8b41aecff): `test_view_default_window_chunks_the_same_at_the_served_context`,
`test_tool_call_arguments_are_the_same_canonical_text`, `test_a_client_that_leaves_a_stream_stops_generation` (native
pairs; skipped for the in-process wasm under Node), `test_decision_after_an_assistant_turn_with_reasoning_scores_the_generation_prompt`;
every pair serves `-c 8192` (`CONTEXT`). `final-wasm-mt` runs by default (`NATLANG_CONFORMANCE_WASM_MT=0` leaves it out);
`NATLANG_CONFORMANCE_WASM_DIR` checks unvendored builds. Five consecutive full runs on fork 8b41aecff (shallow, final,
final-wasm, final-wasm-mt): run 1 with test_serve*.py 140 passed, 2 skipped; runs 2-5 with test_runtime_versions.py on
the vendored builds 116 passed, 2 skipped each (the skips: the client-leaves check on the two in-process wasm params).
`npm run test:browser-neuralese-parity` (ts-host, after `node scripts/build-browser.mjs` refreshes dist/browser): 28/29
gates passed plus one measurement note (exit 0), and 29/29 with `--build neuralese-wasm-mt`; decision scores now agree.

`max_tokens` (2026-10-10, fork 76f7af592, #20): `test_max_tokens_zero_generates_nothing_and_bad_limits_fail`. Both
servers now compute on `NATLANG_CONFORMANCE_THREADS` CPU threads (default min(8, cores): the reference's
`torch.set_num_threads`, the native fork's `-t`; torch's default of every core crawled under machine load). Full run
(shallow, final, final-wasm, final-wasm-mt, with test_serve*.py and test_runtime_versions.py): 146 passed, 2 skipped.
