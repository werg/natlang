# Neuralese port contract

Version 1, 2026-10-03. Normative for runtimes and model servers. Part of the
Neuralese section of [SPEC.md](SPEC.md). The model-side procedures are designed in
[plans/neuralese/S3_PORT.md](../plans/neuralese/S3_PORT.md); the source design is
[port mechanics and training](../plans/neuralese/sources/port-mechanics-and-training.md).

## Control tokens

Two special tokens delimit a soft block:

| Token | LFM2.5 ID slot |
| --- | --- |
| `<|neuralese|>` | `<|reserved_7|>` |
| `<|/neuralese|>` | `<|reserved_8|>` |

Each has its own trained input embedding. Other backbones register the same token
strings as special tokens. A model declares the dialect it reads and writes
([NEURALESE_DIALECTS.md](NEURALESE_DIALECTS.md)).

**Escaping.** Only the runtime (when rendering) and the write procedure (when
writing) produce these token IDs. Renderers tokenize content (messages, tool
outputs, file contents, string values) with special-token parsing disabled, so the
text `<|neuralese|>` in content becomes ordinary text tokens.

## Placement

A block may occur wherever the runtime shows or the model writes TypeScript or
values, inside the backbone's native chat template, with no new role:

- message content, including the opening declarations of a call;
- tool-call arguments, in particular the `code` string of `eval` and the `value` of
  `return_result`. For LFM2.5's Pythonic tool calls
  (`<|tool_call_start|>[eval(code="…")]<|tool_call_end|>`), the block sits inside
  the quoted string. Tool-call parsers recognise the control tokens there and do
  not treat the vectors as string characters;
- tool responses, where eval results and locals are listed.

The block ends at `<|/neuralese|>`. It does not end the message, the tool call, the
string, or the function. Everything that describes a block (its type, a function
signature, captures) is ordinary TypeScript before the opening token.

## Rendering (reference → model form)

When rendering a conversation, the runtime replaces each soft value it shows with
the token `<|neuralese|>`, `L` payload positions, and `<|/neuralese|>`, where `L`
is the stored block's length. It sends the payload as a content part referencing
the block ID (see Wire protocol). The server builds the input embedding sequence
by embedding the surrounding tokens and splicing the stored vectors (mapped into
the model's space by its adapter, if any) at the payload positions. Attention
masks, positions and lengths count every payload position.

## Writing (model form → reference)

When the model samples `<|neuralese|>` during decoding, the server switches to the
write procedure:

1. Run the shallow sketch recurrence (layers `1…k`) from the current context,
   one position at a time. At each position the stop head decides `continue` or
   `stop`; stop is masked before the first position unless empty blocks are
   allowed.
2. On `stop`, run layers `k+1…D` over the collected shallow states in one causal
   pass. The content projection predicts a mean `μ` and a per-dimension scale `σ`
   for every position, and the payload is `z = μ + τ·σ⊙ε` with `ε ~ N(0, I)` at
   the Neuralese temperature `τ` (see Temperature).
3. Store the payload as a new block (content ID, dialect, length, producer) and
   emit `<|/neuralese|>`.
4. Read back: restore the decoder cache to the position before the payload,
   prefill the payload and `<|/neuralese|>` through the full model, and resume
   ordinary decoding.

Restoring the cache covers attention keys and values and short-convolution state,
with positions reset to the committed sequence. Text sampling settings apply only
to text; the block's length is decided by the stop head alone.

**Hard maximum.** The runtime sets a maximum block length per server or request.
Reaching it closes the block, sets `truncated: true` on the stored entry, and is
recorded in the trace; it is not learned stopping.

## Temperature

Writing is stochastic, gated by a **Neuralese temperature** `τ ≥ 0`, separate from
the text sampling temperature. At `τ = 0` the payload is the mean `μ` and writing
is deterministic. At `τ > 0` the payload is a sample `z = μ + τ·σ⊙ε`.

- **Setting.** `τ` is a model-turn setting (`neuraleseTemperature`), defaulting to
  0 at inference. Training sets it on a schedule.
- **Seeds.** `ε` is drawn from a seed derived like other sampling seeds, so replay
  reproduces the same payload.
- **What is stored.** The block's value is the delivered payload `z`, the vectors
  every consumer read. The trace records `μ`, `σ`, `τ` and the seed with the block
  write, so the payload's log-density `log N(z; μ, τ²σ²)` is available to
  `logLikelihood` objectives and replay.
- **Why.** Noise at write time makes representations robust to perturbation
  (VAE-style, with an optional KL term to a standard normal prior in training),
  and gives continuous payloads a tractable likelihood. That lets policy-gradient
  and other sampling-based objectives train encodings, including in RL, alongside
  backpropagation through the payload.

Stored trainable blocks may also be distributions; see
[NEURALESE_FILES.md](NEURALESE_FILES.md).

**Parsing.** In the model turn returned to the runtime, each written block appears
as a content part with its ID, at its position in the text or tool argument. Before
compiling eval code, the runtime replaces each block with the reference expression
`__neuralese.value("nz1_…")` (or `__neuralese.body("nz1_…")` inside an `nl.with`
template), so the TypeScript checker sees ordinary code.

## Wire protocol

Two servers implement this protocol: the reference server
(`training/neuralese/natlang_neuralese/serve/`, PyTorch) and the llama.cpp fork
(`tools/neuralese/neuralese-service.cpp`, also built to WebAssembly for the browser
runtime). Rows marked **both** must agree; `tests/neuralese/test_server_conformance.py`
checks them on identical weights. Rows marked **reference only** or **fork only** are
served by one server; the other answers 501 with the capability's code (Capabilities,
below), never 404 and never by silently ignoring the request.

### Capabilities

`GET /v1/neuralese/info` lists what a server serves in `capabilities`, from this
vocabulary. Clients choose paths by it instead of probing endpoints. A request that needs
a capability the server lacks answers **501** `neuralese-<capability>-unavailable` (dots
become dashes, e.g. `neuralese-chat-stream-unavailable`).

| Capability | Meaning | Reference | Fork (native, wasm) |
| --- | --- | --- | --- |
| `chat`, `decide`, `score`, `render`, `guidance.check`, `encode`, `embed`, `write`, `view`, `template`, `store` | The endpoints and fields of the same names below. | yes | yes |
| `init-body` | `POST /v1/neuralese/init_body`: in-context text initialisation of a soft body with its gate. | yes | no |
| `store.owners` | Owner-scoped holds, pins and collection (`x-natlang-owner`). | listed once its store implements holds | yes |
| `chat.stream` | `"stream": true` on chat completions. | yes | yes (the wasm build streams through its event hook) |
| `template.value-type`, `template.argument-path` | `neuralese_template.value_type` other than `"string"`, `neuralese_template.argument_path`. | yes | yes |
| `parts.value-type` | Content parts with `value_type: "unknown"`. | yes | yes |
| `grad`, `grad.order2` | `POST /v1/neuralese/grad` (order 2: second-order). | yes | no |
| `optim` | `POST /v1/neuralese/optim`. | yes | no |
| `adapters.create` | `POST /v1/neuralese/adapters`. | yes | no |
| `adapters.direct` | Adapter blocks applied directly (`tiny`, `xs`). | yes | no |
| `adapters.projection` | Adapters given as a code through a served projection. | yes | no |
| `adapters.lora-export` | `GET /v1/neuralese/adapters/{id}/lora`. | yes | no |
| `adapters.lora-load` | `PUT /v1/neuralese/adapters/{id}/lora`: adapters apply as loaded LoRAs. | no | yes |

### Conventions

- Bodies are JSON unless stated. Every error answers
  `{"error": {"code": "…", "message": "…"}}`, including unexpected failures (500
  `internal`) and unsupported methods (405 `method-not-allowed`: every method but GET, PUT and POST, HEAD included). A POST body that is not
  a JSON object answers 400 `bad-json` before anything acts on it.
- Status codes: 400 for a request error, including a block the request names that the
  store lacks (`neuralese-unknown-block`), that is in another dialect
  (`neuralese-dialect-mismatch`) or whose width is not the server's (`neuralese-bad-block`); 404 `not-found` for an unknown path and 404
  `neuralese-unknown-block` when fetching or pinning an absent block; 500 for a write
  or decode that failed in the server.
- **Block metadata** ("meta" below) is
  `{"id", "dialect", "length", "width", "dtype", "type"?, "producer"?, "truncated"?}`.
  An endpoint that creates a block stores it before answering; the ID is the content
  ID (`nz1_…`, as `ts-host/src/native/neuralese-store.ts`).
- **Block body**: a safetensors file with one rank-2 `payload` tensor (`F32`, `F16` or
  `BF16`) and the metadata as a JSON string under the `natlang.block` metadata key.

### Content parts

The model-turn request and response (`ts-host/src/contracts.ts`) carry blocks as
content parts, in both directions:

```json
{ "type": "neuralese", "id": "nz1_…", "value_type": "string" }
```

They may appear in message `content` arrays and in tool-call arguments (an argument
string sent as an array of text and neuralese parts, or a JSON argument value that is
such an array). In a response, a part marks a block the server wrote; it is already in
the server's store.

| Field | Meaning | Servers |
| --- | --- | --- |
| `id` | The block. Must start with `nz1_`, else `neuralese-bad-part`. | both |
| `value_type` | `"string"` (default): the block sits inside the quoted string. `"unknown"`: a tool-call argument value that is exactly this block renders unquoted, in native value syntax. A stored block of type `Neuralese<unknown>` defaults to `"unknown"`. | both (`parts.value-type`) |

### Chat completions: `POST /v1/chat/completions`

OpenAI-style chat completion. Request fields beyond OpenAI's:

| Field | Meaning | Servers |
| --- | --- | --- |
| `neuralese_temperature` | `τ` for every block written (see Temperature). Default 0. | both |
| `neuralese_max_length` | Per-request maximum block length, capped by the server's (`max_block_length`). | both |
| `neuralese_length` | Optional size hint: write exactly that many vectors, no stop decision. | both |
| `neuralese_passes` | With a length hint, write the block in that many parallel passes (exact when ≥ the length). | both |
| `neuralese_template` | Template readout: `{"call", "arguments"?, "argument"? (default "value"), "value": "write" \| "decode"}`. The reply is forced to the model's own rendering of that call, cut at the argument; `write` makes it a written block and closes the call, `decode` decodes the value and the rest. Errors: `neuralese-template`. | both |
| `neuralese_template.value_type` | `"string"` (default) or `"unknown"`: the written value sits unquoted; the block is typed `Neuralese<unknown>` and its parts carry `value_type: "unknown"`. Another value is `neuralese-template`. | both (`template.value-type`) |
| `neuralese_template.argument_path` | A list of string or integer keys addressing the value inside nested arguments; it starts at `argument`, and every key must exist in `arguments` (else `neuralese-template`). | both (`template.argument-path`) |
| `x_natlang_adapters` | `[{"id", "scale"}]`: adapter blocks active for the whole request. | both; the fork needs a loaded LoRA per ID, else 409 `neuralese-adapter-not-loaded` |
| `x_natlang_adapters` with `{"code", "projection", "scale"}` | A Neuralese block decoded into an adapter by a served projection (`info.projections`). Errors: `neuralese-projection`. | reference only (`adapters.projection`); the fork answers 501 `neuralese-adapters-projection-unavailable` |
| `guidance` | `true` or `{"require_call"?, "tools"?, "repeat"?, "syntax"?, "retries"?, "run"?}` (`{}`: on, every default; absent, `null` or `false`: off): the reply opens a tool call (`require_call` defaults to `tool_choice == "required"`), call names are checked against `tools` (default: the offered tools), eval code is checked line by line for repetition and TypeScript syntax, and a rejected line is rolled back and resampled. | both; the reference may also apply a server default (`--guidance`) |
| `seed` | Also seeds the payload noise of each written block (with the block's index). | both; the noise generators differ, so payloads at `τ > 0` differ between servers |
| `x_natlang_forced` | Test hook: a plan of text strings and `{"neuralese": "write"}` items that replaces sampling. Errors: `forced-plan`. | both |
| `stream` | `true`: server-sent `chat.completion.chunk` events (below). | both (`chat.stream`) |

Response fields beyond OpenAI's:

| Field | Meaning | Servers |
| --- | --- | --- |
| `choices[0].message` | `content` is an array of text and neuralese parts when the reply wrote a block; tool-call `arguments` is a JSON string whose string values that hold a block are part arrays. That string is canonical, the same bytes on both servers: compact JSON (no spaces after `,` and `:`), non-ASCII characters as UTF-8 rather than `\u` escapes, keys in the order the model wrote them (`json.dumps(value, separators=(",", ":"), ensure_ascii=False)`; the fork's `ordered_json::dump()`). Clients parse it as JSON and do not depend on its spacing. `reasoning_content` holds thinking: a `<think>…</think>` block, or everything before a lone `</think>` (thinking templates open the block in the generation prompt). Tool calls are parsed from Pythonic `<\|tool_call_start\|>[…]<\|tool_call_end\|>` (LFM2) and then JSON `<tool_call>{"name", "arguments"}</tool_call>` markup (the Qwen family: Maple, Mellum), numbered in that order; markup that does not parse stays in `content`. | both |
| `neuralese` | `{"dialect", "blocks": [meta…]}`, one entry per written block, in order. A written block's `producer` is its write record: `{"kind": "write", "request", "index", "cutoff", "temperature", "seed", "stop_logits", "mean"?, "log_sigma"?, "length_hint"?, "passes"?}`, where `mean` and `log_sigma` are stored blocks holding `μ` and `log σ`. A block that hit the hard maximum has `truncated: true`. | both; the fork adds `"rng": "mt19937-normal"` |
| `x_natlang_guidance` | `{"rejections": […]}` when guidance was on. | both |

**Streaming** (both; `chat.stream`). Events are `chat.completion.chunk` objects: a role
delta; text `content` deltas until a tool call opens (call markup, `<|tool_call_start|>` or `<tool_call>`, is held
back, and nothing after it streams as content); per
written block, a delta `{"content": [{"type": "neuralese", "id"}]}` with
`neuralese.block` set to the block's meta; parsed calls as one `tool_calls` delta; a
final chunk with `finish_reason`, `usage`, `neuralese`, `x_natlang_guidance` (when guidance was on), and
`x_natlang_message`, the complete parsed message as a non-streaming response would return it. Clients take the
final message from `x_natlang_message`; the deltas before it are for showing output while it is produced (text
streamed before a guidance rollback is not retracted). An error after
the headers is sent as an `{"error": …}` event (`code` as in a non-streaming answer, `internal` for an unexpected
failure); `[DONE]` ends the stream. A client that disconnects cancels the request: the server notices at its next
event or, while it has nothing to send, between generated positions (the reference polls the connection, the fork
asks `sink.is_writable()`), stops generating and sends nothing more. The WebAssembly service streams through its event
hook (`nzw_handle(…, events)` calls the module's `nzwOnEvent`; a hook that returns `false` cancels the same way), and
the browser's in-process endpoint answers a streamed request at its first event. Under Node the module runs on the
event loop's thread, so the HTTP wrapper cannot see a client leave until the request ends.

### Block store

| Endpoint | Meaning | Servers |
| --- | --- | --- |
| `PUT /v1/neuralese/blocks/{id}` | Store a block (block body). 400 `neuralese-bad-block` for a malformed body, 400 `neuralese-id-mismatch` when the content hashes to another ID. The first block of an ID is kept (a later upload may add a missing `type`). Answers 201 with its meta. | both |
| `GET /v1/neuralese/blocks/{id}` | The block body. | both |
| `GET /v1/neuralese/blocks/{id}/meta` | The block's meta. | both |
| `POST /v1/neuralese/blocks/{id}/pin` | Count one pin on the block for the requesting owner (owners, below; anonymous pins belong to the owner `""`): a pinned block survives every collection, and an owner's pin also holds the block for that owner. 404 `neuralese-unknown-block` for an absent block. Answers `{"ok": true}`. | both |
| `POST /v1/neuralese/blocks/{id}/unpin` | Release one of the requesting owner's pins. Answers `{"ok": true}`. | both |
| `POST /v1/neuralese/collect` | `{"referenced": [id…]}` (anything but a list of IDs is 400 `bad-json`). With an owner: afterwards the owner holds exactly the stored blocks in `referenced` (and those it pins); each block it released is dropped unless another owner holds or pins it. Without an owner: drop every stored block that no owner holds or pins and `referenced` does not name. Answers `{"removed": [id…]}`. | both |

**Owners.** A client names itself (a session or runtime ID) in the `x-natlang-owner` request header. The owner holds
every block it uploads (`PUT`) and every block ID a successful response names to it (a written block, a fetched block,
a gradient); pins and collections are per owner, so one session's collection never drops another session's blocks.
A client that receives `neuralese-unknown-block` for a block it referenced (the server restarted or the block was
collected) uploads the block again from its own content-addressed store and retries. Whether blocks, holds and pins
survive a server restart is a server option (the reference's `--store-dir`), not part of the wire protocol; `info`
reports it. The WebAssembly service has no request headers and serves a single, anonymous owner. In a browser the
client's own store is the runtime's OPFS block archive (`OpfsNeuraleseStore`, `ts-host/src/browser/neuralese-opfs-store.ts`):
one file per block in the origin-private file system, named by its ID and holding the block's safetensors body (as
`PUT` sends it), its metadata indexed when the archive opens and its bytes checked against the ID when read. The in-page
service keeps blocks only in memory, so after a page reload or an engine restart the runtime restores every block a
request names from that archive. Blocks move between the archive and `.nz` files by ID (`importNz`, `exportNz`).

### Writing blocks

Each answers 201 with the new block's meta.

| Endpoint | Meaning | Servers |
| --- | --- | --- |
| `POST /v1/neuralese/write` | The write procedure at a write site: `{"messages", "prefix"?, "tools"?, "neuralese_temperature"?, "length"?, "passes"?}`. The reply is forced to `prefix` and then the open marker; the stop head decides the length unless `length` hints it (`passes` as `neuralese_passes`). 500 `neuralese-write` if no block was written. | both |
| `POST /v1/neuralese/encode` | Text into a block in one forward pass through the port (supplied-input write, one vector per token, no stop decision): `{"text", "type"?, "context"?}`, where `context` is chat messages without blocks rendered as the write site. Errors: `neuralese-encode`. Producer `{"kind": "text-encode", "text"}`. | both |
| `POST /v1/neuralese/embed` | A block initialised from the token embeddings of `{"text", "type"?}`. Errors: `neuralese-embed`. Producer `{"kind": "text-init", "text"}`. | both (`embed`) |
| `POST /v1/neuralese/init_body` | In-context text initialisation of a soft body (`natlang_neuralese/text_init.py`): `{"messages", "tools"?, "placeholder", "text", "type"?, "gate"?, "reply_tokens"?}`, where `messages` is the soft call as the runtime renders it with the block `placeholder` at the body's position (exactly once). The body's rows are the read transport's inverse of the embeddings of the instruction tokens as the text-instructed call (the same rendering with `text` in place of the placeholder) tokenizes them; exact when the transport has no markers and an identity input map and the span tokenizes as in the joint rendering (`aligned`). With `gate` (default true) the soft call with the body is compared with the text-instructed call over the generation position and a greedy reply of `reply_tokens` (24) tokens: `{"agreement", "kl", "max_abs_logit_delta", "bit_exact", "passed"}` (passed: agreement ≥ 0.98, KL ≤ 0.02). Answers 201 `{"block", "init", "gate"}`. Errors: `neuralese-init-body`. Producer `{"kind": "text-init-in-context", "text", …init}`. | reference only (`init-body`) |
| `POST /v1/neuralese/view` | The Neuralese instance of the builtin `view(value, instructions?)` (`natlang_neuralese/view.py`, pinned by `tests/fixtures/view-site.json`): `{"value", "instructions"?, "system"?, "window"?}`. Every write is the template write of view's body at its site: the system text is the body (`system`, as text or parts with its soft form; default the body's text), the user message gives `instructions` (if any) and the value, and the reply is forced by template readout to `return_result(status="success", value=…)` with the value written as a block (the site offers that one tool). Without `instructions` the view is faithful compression. Answers the view block's meta plus `parts` (1 unless the value exceeds the write site's window and is viewed in chunks, then combined at a combine site that reads the part views) and `window`. The default window is what the writer attends to at the write site: the serving server's `context` (info) less the site's prompt rendered without the value (system text or parts, the user text with any instructions, the tool, the chat template), the reply (`max_block_length` + 64: the block and the forced call) and 32 tokens (a part site's longer header, retokenized chunk edges), at least 256; a request `window` can only lower it. Two servers serving the same context plan the same chunks. | both |

### Readouts

| Endpoint | Meaning | Servers |
| --- | --- | --- |
| `POST /v1/neuralese/decide` | Decision readout: `{"messages", "options", "tools"?, "adapters"?}` → `{"log_probs", "tokens"}`. Each option is a reply text scored as the whole assistant reply after one prompt pass; `tokens[i]` counts the tokens where the options differ. Errors: `neuralese-decision`. `adapters` as `x_natlang_adapters`. | both |
| `POST /v1/neuralese/decide_many`, `POST /v1/natlang/score` | Batched decisions (plans/BATCHED_EXECUTION.md): `{"items": [{"messages", "options" \| "continuations", "tools"?, "adapters"?}], "adapters"?}` → `{"results": [{"log_probs", "tokens"} \| {"error"}]}`, one per item in order. Each result equals `decide` on that item alone; items with the same adapters, messages and tools share one prefill wherever they sit in the request (servers group them and answer in request order); a failing item fails alone. Top-level `adapters` is the default for items. An item's `error` is `"<code>: <detail>"` for a request error (the same text on both servers) and `"internal: …"` otherwise. | both |
| `POST /v1/neuralese/render` | The rendered prompt of `{"messages", "tools"?}` with each block as `<block>` → `{"prompt"}` (201). For conformance. | both |
| `POST /v1/neuralese/guidance/check` | `{"reply", "guidance": {"tools"?, "repeat"?, "syntax"?, "run"?}}` → the first rejection when `reply` is checked prefix by prefix as during generation, `{"reason", "offset", "end"}`, or `{"reason": null}` (201). For conformance. | both |

### Learning

| Endpoint | Meaning | Servers |
| --- | --- | --- |
| `POST /v1/neuralese/grad` | Gradient replay session: `{"arguments": [id…], "terms": [term…], "producers"?, "adapters"?, "order"? (1 or 2), "derived"?}` → `{"loss", "terms": [loss…], "gradients": {argument id: gradient id}}`. Term kinds: `crossEntropy`, `logLikelihood`, `decision`, `selfDistill`, `klPrior`. Gradient blocks are in dialect `{dialect}#grad`. Errors: `neuralese-grad-term`, `neuralese-grad-target`, `neuralese-grad-derived`, `neuralese-grad-unavailable`. | reference only (`grad`, `grad.order2`); the fork answers 501 `neuralese-grad-unavailable` |
| `POST /v1/neuralese/optim` | One optimiser step: `{"optimizer": "sgd" \| "adam", "hyper", "params": [id…], "grads": [id…], "state"?: {"step", "m"?, "v"?}}` → `{"params": [id…], "state": {"step", "m"?, "v"?}}`. State blocks are in dialect `{dialect}#opt`. Errors: `neuralese-optim`. | reference only (`optim`); the fork answers 501 `neuralese-optim-unavailable` |

Gradient determinism (reference server setting): on a GPU the attention backward accumulates with atomics, so the same `grad` request can return different gradient bytes (and, after a learning-rate step, different losses). `NATLANG_NEURALESE_DETERMINISTIC=1` or `python -m natlang_neuralese.serve --deterministic-gradients` runs `GradSession.run` under strict `torch.use_deterministic_algorithms` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (`GradSession(engine, deterministic=…)` per session). Off by default: it is slower on long contexts and an op without a deterministic CUDA kernel fails the request, so trainers sharing `GradSession` enable it only when a recipe declares it. The learning tests and the conformance suite set it. Order 1 and order 2 use different kernels (order 2: math attention, unfused convolution), so under bf16 their losses for one request agree to a few percent, not exactly.

### Weight adapters

Adapter blocks (`model/tiny_adapters.py`) bind through `x_natlang_adapters` and
`adapters`. The reference applies them directly; the fork applies each as a GGUF LoRA
that a client loads first.

| Endpoint | Meaning | Servers |
| --- | --- | --- |
| `POST /v1/neuralese/adapters` | A zero adapter for this backbone: `{"kind", "rank", "u", "layers", "targets", "seed", "type"?}` → meta (201). | reference only (`adapters.create`); the fork answers 501 `neuralese-adapters-create-unavailable` |
| `GET /v1/neuralese/adapters/{id}/lora` | A stored adapter exported as a GGUF LoRA (octet stream). 404 for a block that is not a stored adapter. | reference only (`adapters.lora-export`); the fork answers 501 `neuralese-adapters-lora-export-unavailable` |
| `PUT /v1/neuralese/adapters/{id}/lora` | Load a GGUF LoRA body as adapter `{id}` → `{"id", "loaded": true}` (201). 400 `neuralese-adapter-lora` if it is not a LoRA for this model. | fork only (`adapters.lora-load`); the reference applies adapters directly and answers 501 `neuralese-adapters-lora-load-unavailable` |

### Info and health

`GET /v1/neuralese/info`:

| Field | Meaning | Servers |
| --- | --- | --- |
| `dialects` | Dialects spoken (one per server today). | both |
| `width`, `dtype`, `max_block_length`, `cutoff` | Payload width, payload dtype (`f32`), hard maximum block length, shallow cutoff layer. | both |
| `context` | The served context in tokens: positions one request's prompt and reply may span. Fork: `min(n_ctx_train, -c)` (default `-c` 8192); reference: `min(max_position_embeddings, -c)` (default the model's `max_position_embeddings`). `view` plans its write sites against it. | both |
| `grad` | Whether `grad` sessions are served: `true` (reference), `false` (fork). | both |
| `grad_order` | Highest gradient order: 2 (reference), 0 (fork: none). | both |
| `adapters` | Adapter kinds applied directly: `["xs", "tiny"]` (reference), `[]` (fork: adapters apply as LoRAs loaded with `PUT …/lora`, capability `adapters.lora-load`). | both |
| `projections` | `{name: {"source", "target", "identity"}}`: projections that decode adapter codes (`{}` on the fork). | both |
| `stream` | `true`: chat completions honour `"stream": true` (Streaming, above). Clients stream only to a server that declares it; absent means not streaming. | both |
| `store` | `{"owners": true, "persistent": bool}`: owner-scoped holds, pins and collection; whether blocks outlive a restart. | both |
| `server` | `"reference"` or `"llama.cpp"`. | both |
| `capabilities` | The sorted list of capabilities served (Capabilities, above). Clients read this rather than inferring from `adapters` or `grad`. | both |

`GET /health`, `GET /v1/health` (`{"status": "ok"}`) and `GET /v1/models` are served
by both.

### No text fallback

A request containing neuralese parts sent to a server that does not declare a
matching dialect fails; the runtime reports `neuralese-unsupported-backend` or
`neuralese-dialect-mismatch`. There is no text fallback.

## Agreement

A server's cached execution must agree with recomputation on the same committed
sequence: identical greedy continuations and logits within the tolerance stated by
the parity suite (`conformance/neuralese/`).
