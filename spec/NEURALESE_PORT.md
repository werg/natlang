# Neuralese port contract

Version 2, 2026-10-26. Normative for runtimes and model servers. Part of the
Neuralese section of [SPEC.md](SPEC.md). The design and its cutover are tracked in
[plans/neuralese/SHARED_STATE_MIGRATION_20261015.md](../plans/neuralese/SHARED_STATE_MIGRATION_20261015.md).
The projected-value protocol (vector payload blocks, `nz1_` IDs and their endpoints)
is retired; servers reject its request fields and content parts.

Neuralese is native shared state. A producer writes ordinary tokens between producer
markers; the server captures the positional KV and hybrid layer state of that span
and stores it as a native state. A consumer imports a stored state between consumer
markers, and the server splices it into the consumer's cache at the consumer's own
positions. No separate communication latent space exists.

## Role markers

Four special tokens mark producing and consuming:

| Marker | Meaning |
| --- | --- |
| `<\|neuralese_produce\|>` | Opens a producer span. |
| `<\|/neuralese_produce\|>` | Closes a producer span. |
| `<\|neuralese_consume\|>` | Precedes an imported native state. |
| `<\|/neuralese_consume\|>` | Follows an imported native state. |

A native checkpoint adds the four strings as special tokens with four distinct IDs.
The server reports the IDs in this order as `native_state.role_marker_ids` in
`GET /v1/neuralese/info`.

**Embeddings.** Producer tokens are ordinary vocabulary tokens with their ordinary
input embeddings. The model applies a learned low-rank residual transform, inside
embedding space, to the embeddings of positions that are producing. The transform
starts at the identity.

**Escaping.** Only the server inserts marker token IDs. When it renders a request,
every special-token string that occurs inside message content or tool-call arguments
is tokenized as ordinary text, so the text `<|neuralese_produce|>` in content never
becomes a marker. Only structure inserted by the chat template becomes special tokens.

## Producing

A producer span is opened and closed by the model's own sampled tokens, or by a
forced template write (Template readout, below).

1. When the model samples `<|neuralese_produce|>`, the server feeds it as an
   ordinary input and opens a span.
2. Each following token is fed as a producing input: the low-rank transform applies
   to it, and its per-layer state is captured.
3. When the model samples `<|/neuralese_produce|>`, the server feeds it as an
   ordinary input, closes the span, stores the captured state as a new native state,
   and continues decoding.

The markers and the span's tokens do not appear in the reply text. The reply carries
a `native_state` content part at the span's position instead.

The server rejects these draws with the listed code:

| Draw | Code |
| --- | --- |
| `<\|/neuralese_produce\|>` with no open span | `neuralese-native-state-close` |
| `<\|neuralese_consume\|>` or `<\|/neuralese_consume\|>` at any point | `neuralese-native-state-consume-marker` |
| `<\|neuralese_produce\|>` inside an open span | `neuralese-native-state-nested` |
| A stop token inside an open span, or `max_tokens` reached inside one | `neuralese-native-state-open` |

Without protocol recovery, the request fails with that code. With it, the server
answers a failure observation (Native marker failure observations, below).

There is no separate span length limit in chat: a span ends when the model closes it,
and the request's `max_tokens` bounds it.

## Consuming

A native state referenced in a request renders as `<|neuralese_consume|>`, the
imported state, and `<|/neuralese_consume|>`. The server chooses how to import the
state, in this order:

| Mode | When | What the server does |
| --- | --- | --- |
| `same-base-adapter-state` | The request declares this mode for the state's ID (`x_natlang_native_state_modes`). | Splices the raw state, validated against its own source identity on the same base checkpoint. The state is not relabelled. |
| `exact-native-state` | The state's checkpoint (as executed, including active adapters), tokenizer fingerprint, embedding-weight digest, architecture and layer layout equal the consumer's. | Splices the stored state at the consumer's positions. |
| `learned-semantic-state-deltas` | The server runs with a semantic adapter whose bindings include both the state's identity and the consumer's. | Adapts the state through the adapter, then splices the result. |
| `best-effort-text-retokenization` | Otherwise. | Discards the stored tensors, tokenizes the state's text with the consumer's tokenizer, and prefills those tokens between the markers. |

Attention keys are stored before rotary embedding, so the consumer rotates them at
its own logical positions. An exactly imported state occupies its token count plus
the two markers in the consumer's positions.

Each import is reported in the response's `native_state.imports`:
`{"id", "mode", "observed_native_state_id"?, "semantic_adaptation"?,
"adapter_state_consumption"?, "state_derivation"?, "stop_gradient"?,
"source_occurrence"?}`. `observed_native_state_id` names the state actually spliced
(absent for text retokenization).

Errors: `neuralese-unknown-native-state` (the store lacks the state),
`neuralese-native-state-mode` (an invalid mode declaration),
`neuralese-semantic-adapter-mismatch` (semantic adaptation failed),
`neuralese-native-state-empty-text` (text retokenization of a state with tokens but
no text).

## Native states

A native state (`schema: "neuralese.native-state/1"`) holds:

- `text`: the span's source text, and `tokenizer_fingerprint`, `checkpoint_sha256`,
  `embedding_weight_sha256`, `architecture`: the producer's identity;
- tensors `token_ids` (int64), `positions` (int64, contiguous), `embeddings` (the
  tokens' input embeddings), all with the token count as first dimension;
- one state per layer, listed in `layer_kinds`: `attention` (`raw_k`, `v`: pre-rotary
  keys and values), `finite_convolution` (`gated_inputs`), or `gated_delta`
  (`mixed_qkv`, `z`, `b`, `a`);
- optional `adapter_binding` (the actor adapters active when it was produced) and
  `derivation` (a state produced by an optimizer step).

A server with a semantic adapter stores produced spans as semantic messages
(`schema: "neuralese.semantic-state/1"`) carrying `adapter_sha256` and a code tensor.

Reuse of raw state requires the exact identity above. Moving a state to another
checkpoint, tokenizer or architecture goes through text retokenization or a learned
semantic adapter; equal tensor shapes do not make states interchangeable.

**ID.** `nzs1_` followed by the lowercase, unpadded base32 SHA-256 (52 characters) of:
the bytes `natlang.native-state/1`, a NUL byte, the metadata length as a
little-endian u64 and the metadata as canonical JSON (sorted keys, no spaces,
non-ASCII as UTF-8); then for each tensor in name order, the name length (u32) and
name, the descriptor `{"dtype","shape"}` length (u32) and JSON, and the raw byte
length (u64) and raw little-endian bytes.

**Body.** A safetensors file with one entry per tensor (`F16`, `BF16`, `F32`, `F64`
or `I64`), payloads contiguous in name order without gaps, and under the
`__metadata__` key `natlang.native_state` the metadata with its `id` as a canonical
JSON string. A decoder rejects gaps, overlaps, unknown tensors and an `id` that does
not match the content.

**Native tensor artifacts** (`nnt1_…`, same ID construction under the domain
`natlang.neuralese-native-tensor-artifact/1`) hold named tensors with metadata
`{"schema", "kind", "native_identity", "source_artifact_id", "provenance",
"tensors"}`. Kinds: `parameter-set`, `parameter-gradient`, `parameter-update`,
`state-gradient`, `optimizer-state`, `adapter-basis`, `adapter-coefficients`,
`adapter-gradient`, `derivative-program`. The body metadata key is
`natlang.native_tensor_artifact`.

## Wire protocol

This repository contains one implementation of this protocol, the reference server
(`training/neuralese/natlang_neuralese/serve/`, `python -m natlang_neuralese.serve`).
The vendored WebAssembly build in `ts-host/vendor/neuralese-wasm/` (the browser's local
model engine) predates native state: it serves chat completions, decisions and guidance,
and declares no `native_state` in `info`, so a runtime configured for Neuralese values
refuses it at setup (`neuralese-native-state-unavailable`).

### Capabilities

`GET /v1/neuralese/info` lists what the server serves in `capabilities`. Clients choose
paths by it instead of probing endpoints. The TypeScript client fails a request that
needs an unlisted capability with `neuralese-<capability>-unavailable` (dots become
dashes) before sending it.

| Capability | Meaning |
| --- | --- |
| `chat`, `chat.stream` | Chat completions; `"stream": true`. |
| `template`, `template.value-type`, `template.argument-path` | `neuralese_template` and its `value_type` and `argument_path` fields. |
| `parts.value-type` | Template-written parts carry `value_type` in tool-call arguments. |
| `guidance`, `guidance.check` | The `guidance` request field; `POST /v1/neuralese/guidance/check`. |
| `decide`, `score` | `POST /v1/neuralese/decide`; `POST /v1/neuralese/decide_many` and `POST /v1/natlang/score`. |
| `render` | `POST /v1/neuralese/render`. |
| `view` | `POST /v1/neuralese/view`. |
| `native.text-write`, `native.init-body` | `POST /v1/neuralese/native-states/write-text`, `/init-body`. |
| `native.protocol-recovery` | The `x_natlang_native_protocol_recovery` request field. |
| `grad`, `grad.order2` | `POST /v1/neuralese/grad`. A graph request with `order` other than 1 is rejected; differentiation through optimizer steps uses derivative programs. |
| `optim` | `POST /v1/neuralese/optim`. |
| `native.adapters`, `adapters.create` | Exact-base actor adapters: `x_natlang_adapters` and `adapters` fields; `POST /v1/neuralese/adapters`. |
| `store`, `store.native-state`, `store.native-tensor-artifact` | The native state and tensor artifact routes. |
| `store.owners` | Owner-scoped holds, pins and collection (`x-natlang-owner`). |

### Conventions

- Bodies are JSON unless stated. Errors answer `{"error": {"code": "…", "message":
  "…"}}`; for a request error the message is `"<code>: <detail>"`.
- A POST body that is not valid JSON or not a JSON object answers 400 `bad-json`.
- 400 for a request error, with the error's code. 404 `not-found` for an unknown path.
  405 `method-not-allowed` for every method but GET, PUT and POST, HEAD included.
  500 `internal` for an unexpected failure. Every failure of every method answers this
  envelope ([ERRORS.md](ERRORS.md) "Model server codes").
- The server reads only the request fields named in this document. Other fields are
  ignored, except the retired fields listed under Chat completions, which are rejected.
- An endpoint that creates a native state or tensor artifact stores it before answering.

### Content parts

Messages carry native states as content parts, in requests and responses:

```json
{ "type": "native_state", "id": "nzs1_…" }
```

| Field | Meaning |
| --- | --- |
| `id` | `nzs1_` and 52 base32 characters, else 400 `neuralese-bad-part`. |
| `stop_gradient` | Optional; must be `true` when present (else `neuralese-bad-part`). The import is a constant in gradients and captures. |
| `source_occurrence` | Optional non-empty string naming the producer occurrence (else `neuralese-bad-part`). Copied to the import receipt and capture. |
| `value_type` | Emitted by the server on parts in a template-written argument: `"string"` or `"unknown"`. Ignored on input. |

Parts occur in message `content` arrays (with `{"type": "text", "text"}` parts) and in
tool-call arguments: an `arguments` value that is a part array, or a JSON argument
value that is such an array. An argument value that is exactly one native state
renders unquoted in the template's call syntax; a state inside a longer string stays
inside the quoted string. Inside a nested argument value, a part array with a
malformed `native_state` ID is kept as ordinary argument data.

A part of type `"neuralese"` anywhere in content or arguments answers 400
`neuralese-retired-part`. A chat template that drops or duplicates a reference
answers 400 `neuralese-render`. Tool-call `arguments` given as JSON text that does not
parse answer 400 `tool-arguments`.

### Chat completions: `POST /v1/chat/completions`

OpenAI-style chat completion. Request fields:

| Field | Meaning | Errors |
| --- | --- | --- |
| `messages`, `tools` | As OpenAI, with content parts. | |
| `max_tokens`, `max_completion_tokens` | Generation allowance, first present one; default 512; 0 generates nothing (`finish_reason: "length"`). Counts every generated position, markers and producer tokens included. | `bad-max-tokens` for anything but a non-negative integer |
| `temperature` | 0 or absent: greedy. Above 0: sampling at that temperature. | |
| `seed` | Seeds sampling. Default: derived from the request ID. | |
| `tool_choice` | Read only by guidance (`require_call` default). | |
| `stream` | `true`: server-sent events (Streaming). | |
| `guidance` | `true` or `{"require_call"?, "tools"?, "repeat"? (3), "syntax"? (true), "run"? (4), "retries"? (4)}`; absent, `null` or `false`: off, unless the server was started with `--guidance`. The reply opens a tool call when `require_call` (default `tool_choice == "required"`); call names are checked against `tools` (default: the offered tools); eval code is checked line by line for repetition, redeclaration and TypeScript syntax; a rejected line is rolled back to its exact cache state and resampled with the offending token banned, at most `retries` times per point. A native state in eval code is read as the identifier `__nz`. | `neuralese-guidance-template-conflict` when a forced template violates guidance |
| `neuralese_template` | Template readout: `{"call", "arguments"? ({}), "argument"? ("value"), "value": "write" \| "decode", "value_type"? ("string" \| "unknown"), "argument_path"?}`. The reply is forced to the model's own rendering of that call, cut at the argument's value. `write`: the server forces `<\|neuralese_produce\|>`, the model writes the span until it closes it, and the server forces the rest of the call. `decode`: the value and the rest are generated as text. `value_type: "unknown"` places the value unquoted. `argument_path` is a list of string or integer keys addressing the value inside nested arguments, starting at `argument`. | `neuralese-template` |
| `x_natlang_adapters` | Actor adapters for this request: a list of `nnt1_` coefficient artifact IDs or `{"id", "scale"?, "stop_gradient"?}`. | `neuralese-native-adapter-invalid` |
| `x_natlang_native_state_modes` | `{state_id: "same-base-adapter-state"}` for states named in the request. | `neuralese-native-state-mode` |
| `x_natlang_rl_capture` | `true`: return `x_natlang_rl_capture`. Requires `temperature` above 0 unless `x_natlang_native_recording` is set. | `native-rl-sampling-law` |
| `x_natlang_native_recording` | `true`: structural capture, at any temperature; also returns `x_natlang_rl_capture`. | |
| `x_natlang_forced_message` | Teacher-forced replay: `{"schema": "natlang.native-teacher-forced-turn/1", "source_receipt", "assistant", "native_text_spans"?}`. The reply is forced to exactly the assistant message; each `native_text_spans` entry (`{"path", "marker", "text"}`) becomes a producer span at its marker. Requires `x_natlang_native_recording: true`. | `bad-request`, `forced-message` |
| `x_natlang_native_protocol_recovery` | Boolean. `true`: marker violations answer a failure observation instead of an error. | `bad-request` if not boolean |

Retired fields answer 400 `unsupported-request-field`: `x_natlang_forced`,
`neuralese_temperature`, `neuralese_max_length`, `neuralese_length`,
`neuralese_passes`.

Other request errors: `empty-prompt`, `prompt-rendering` (the messages do not fit the
server's declared prompt rendering), `context-length` (the prompt, or a template
prefix, does not fit the served context), `neuralese-native-actor-changed` (the adapter
binding changed during a captured request), `cancelled` (the client left).

Response fields beyond OpenAI's:

| Field | Meaning |
| --- | --- |
| `id` | `chatcmpl-<n>`. Tool-call IDs are `<id>_<index>`. |
| `choices[0].message` | `content` is a string, `null`, or an array of text and `native_state` parts when the reply produced a state. Tool-call `arguments` is compact JSON text (no spaces after `,` and `:`, non-ASCII as UTF-8, keys in the order the model wrote them); a string value that holds a state is a part array. `reasoning_content` holds a `<think>…</think>` block, or everything before a lone `</think>`. Tool calls are parsed from Pythonic `<\|tool_call_start\|>[…]<\|tool_call_end\|>` (LFM2) and then JSON `<tool_call>{"name", "arguments"}</tool_call>` (Qwen family), numbered in that order; markup that does not parse stays in `content`. |
| `choices[0].finish_reason` | `stop`, `length`, or `error` (a failure observation). |
| `usage` | `prompt_tokens`: prompt positions, imports and their markers included. `completion_tokens`: generated positions, markers and producer tokens included. |
| `native_state` | `{"blocks": [meta…], "imports": [receipt…]}`: every state the reply produced, in order, and every import (Consuming). |
| `x_natlang_guidance` | `{"rejections": [{"reason", "offset", "accepted"?}]}` when guidance was on. |
| `x_natlang_rl_capture` | When requested, recorded, or on a failure observation: `{"schema", "policy", "adapter_binding"?, "teacher_source_receipt"?, "replay_prompt_sha256"?, "prompt", "samples", "generated_inputs", "writes", "finish_reason", "sample_range_convention": "half-open", "sampling_law": {"temperature", "seed", "guidance", "probabilities"}, "protocol_failure"?}`. Schemas: `natlang.native-teacher-forced-capture/1` (forced replay), `natlang.native-structural-capture/1` (temperature 0), `natlang.native-rl-sampling/3` (with adapters) or `/2` (sampled); on a failure observation `natlang.native-structural-capture/2` or `natlang.native-rl-sampling/4`. `prompt` lists token segments (with per-token `producing` flags) and imported states; `writes` gives each produced state with its sample and own-position ranges. |
| `x_natlang_native_failure` | The failure descriptor (Native marker failure observations). |

**Streaming** (`chat.stream`). Events are `chat.completion.chunk` objects sent with
chunked transfer encoding as `text/event-stream`: a role delta; text `content` deltas
until tool-call markup (`<|tool_call_start|>` or `<tool_call>`) opens, after which
nothing streams as content; per produced state a delta
`{"content": [{"type": "native_state", "id"}]}` with `native_state.block` set to its
meta; parsed calls as one `tool_calls` delta; a final chunk with `finish_reason`,
`usage`, `x_natlang_message` (the complete message as a non-streaming response returns
it), and `native_state`, `x_natlang_rl_capture`, `x_natlang_guidance`,
`x_natlang_native_failure` when present. Producer-span tokens never stream as text.
With guidance on, deltas are sent only for committed output when generation ends.
Clients take the final message from `x_natlang_message`. An error after the headers is
an `{"error": …}` event (`internal` for an unexpected failure); `[DONE]` ends the
stream. A client that disconnects cancels the request: the server checks the
connection while it has nothing to send and stops generating.

### Native artifact store

| Endpoint | Meaning |
| --- | --- |
| `PUT /v1/neuralese/native-states/{id}` | Store a native state (body, `application/octet-stream`). 201 with its meta. 400 `neuralese-bad-native-state` for a body that does not decode or that the store refuses (a failed derivation check, a content collision), 400 `neuralese-id-mismatch` when the content hashes to another ID. A repeated upload of identical content is accepted. |
| `GET /v1/neuralese/native-states/{id}` | The body. 404 `neuralese-unknown-native-state`. |
| `GET /v1/neuralese/native-states/{id}/meta` | `{"id", …metadata}`. 404 `neuralese-unknown-native-state`. |
| `PUT /v1/neuralese/native-tensor-artifacts/{id}` | Store a tensor artifact. 201 with its meta. 400 `neuralese-bad-native-artifact` for a body that does not decode or that the store refuses (a failed gradient-origin check, a content collision), 400 `neuralese-id-mismatch`. |
| `GET /v1/neuralese/native-tensor-artifacts/{id}`, `…/{id}/meta` | Body or meta. 404 `neuralese-unknown-native-artifact`. |
| `POST /v1/neuralese/native-states/{id}/pin`, `POST /v1/neuralese/native-tensor-artifacts/{id}/pin` | Count one pin for the requesting owner (anonymous pins belong to the owner `""`). A pinned artifact survives every collection; an owner's pin also holds it for that owner. `{"ok": true}`. 404 `neuralese-unknown-native-state` or `neuralese-unknown-native-artifact` for an absent artifact. |
| `POST …/{id}/unpin` | Release one of the requesting owner's pins. `{"ok": true}`. |
| `POST /v1/neuralese/collect` | `{"referenced": [id…]}` (absent or `null`: empty; anything but a list of strings is 400 `bad-json`). With an owner: afterwards the owner holds exactly the stored artifacts in `referenced` and those it pins; each artifact it released is removed unless another owner holds or pins it. Without an owner: remove every stored artifact that no owner holds or pins and `referenced` does not name. Artifacts that a kept, held or pinned artifact depends on (adapter bases, derivation inputs) are kept. `{"removed": [id…]}`. |

**Owners.** A client names itself (a session or runtime ID) in the `x-natlang-owner`
request header; an absent or empty header is anonymous. An owner holds every artifact
it uploads, every artifact a request of its own produces (including states produced
before a failure), and every `nzs1_`/`nnt1_` ID that appears as a whole string in a
successful JSON response or stream event sent to it. Pins and collections are per
owner, so one session's collection never removes another session's artifacts. A
client that receives `neuralese-unknown-native-state` or
`neuralese-unknown-native-artifact` for an artifact it referenced uploads it again
from its own archive and retries.

Whether artifacts, holds and pins survive a restart is a server option
(`--store-dir`): the store then keeps one safetensors file per artifact and an SQLite
index on disk, with a bounded resident set (`--store-resident-gb`, default 2).
`info.store.persistent` reports it. One process owns a store directory.

### Producing states outside chat

| Endpoint | Meaning |
| --- | --- |
| `POST /v1/neuralese/native-states/write-text` | `{"text", "type"?, "adapters"?}`. Runs `<\|neuralese_produce\|>`, the text's tokens as producing inputs, and `<\|/neuralese_produce\|>` from an empty context, and stores the span. 201 `{"id", "native_state": meta, "source_capture": {"policy", "segments", "payload_start", "payload_stop"}, "source_text_sha256"}`; `type`, when given, is copied into the returned meta only. Errors: `neuralese-native-state-text`, `context-length`. |
| `POST /v1/neuralese/native-states/init-body` | Contextual text initialisation of a soft body: `{"messages", "tools"?, "placeholder", "text", "type"?, "placeholder_occurrence"?, "adapters"?, "native_state_modes"?}`. `placeholder` is a stored `nzs1_` state referenced by a `native_state` part in `messages` (also inside JSON-text tool arguments); the selected occurrence is replaced by `text` wrapped in producer markers at its rendered position, the rest of the prompt (other imports included) is run, and the body's state is captured. Without `placeholder_occurrence` the placeholder must occur exactly once. 201 `{"id", "native_state", "init": {"schema": "natlang.native-state-text-init/1", "tokens", "exact": false, "aligned": true, "exact_transport": true, "contextual": true, "source_position_start"}, "gate": null, "source_capture": {…, "imports"}, "source_text_sha256"}`. Errors: `neuralese-native-state-text`, `neuralese-unknown-native-state`, `neuralese-native-state-placeholder`, `neuralese-native-state-mode`, `context-length`. |
| `POST /v1/neuralese/view` | The Neuralese instance of the builtin `view(value, instructions?)` (`natlang_neuralese/view.py`): `{"value", "instructions"?, "system"?, "window"?, "x_natlang_native_protocol_recovery"?}`. Each write is a template write of `return_result(status="success", value=…)` at the view site, sampled at temperature 1 with capture: the system message is view's body (`system`, as text or parts; default the body's text), the user message gives `instructions` (if any) and the value. Without `instructions` the view is faithful compression. A value that exceeds the window is viewed in chunks and combined. 201 with the final state's meta plus `"parts"` (number of chunks, 1 when unchunked), `"window"` and `"source_turns"` (every write's messages, tools, template and full response). The default window is the served `context` less the site's prompt without the value and less `max_block_length` + 96 reserved tokens; a request `window` can only lower it. On a failure observation: 200 `{"x_natlang_native_failure", "source_turns", "window"}`. Errors: `neuralese-view-window`, `context-length`, `neuralese-view`, `neuralese-view-source`. |

### Readouts

| Endpoint | Meaning |
| --- | --- |
| `POST /v1/neuralese/decide` | `{"messages", "options", "tools"?, "adapters"?, "native_state_modes"?, "native_recording"?}` → 200 `{"log_probs", "tokens", "native_state_imports", "native_decision_capture"?}`. Each option is a reply text scored as the whole assistant reply, end of message included, after one shared prompt pass; `tokens[i]` is the option's token count. `native_recording` adds `native_decision_capture` (`natlang.native-decision-capture/1`). Errors: `neuralese-decision` (no options, non-string options, options that render different prompts or contain states). |
| `POST /v1/neuralese/decide_many`, `POST /v1/natlang/score` | `{"items": [{"messages", "options" \| "continuations", "tools"?, "adapters"?, "native_state_modes"?, "native_recording"?}], "adapters"?, "native_state_modes"?, "native_recording"?}` → 200 `{"results": [{"log_probs", "tokens", …} \| {"error"}]}`, one per item in order. Top-level fields are item defaults. Each result equals `decide` on that item alone; a failing item fails alone with `error` `"<code>: <detail>"` (`internal` for an unexpected failure), cut to 500 characters. `items` not a list: 400 `neuralese-decision`. |
| `POST /v1/neuralese/render` | `{"messages", "tools"?}` → 201 `{"prompt"}`: the rendered prompt with each native-state reference as `<block>`. |
| `POST /v1/neuralese/guidance/check` | `{"reply", "guidance": {"tools"?, "repeat"?, "syntax"?, "run"?}}` → 201 with the first rejection when `reply` is checked prefix by prefix as during generation, `{"reason", "offset", "end"}`, or `{"reason": null}`. Reasons: `unknown-tool`, `repetition`, `redeclaration`, `syntax`. |

### Learning

| Endpoint | Meaning |
| --- | --- |
| `POST /v1/neuralese/grad` | One of four request forms. (1) `{"schema": "natlang.native-gradient-request/1", "graph", "target", "objective"? ("graph-ce"), "native_identity"?, "graph_sha256"?, "adapters"?, "invocation_adapters"?, "native_state_modes"?, "adapter_gradient_precision"?}`: one complete train-split native graph row evaluated in one pass; `target` is `{"kind": "model-parameters", "names"}`, `{"kind": "adapter", "id", "names"?}` or `{"kind": "native-state", "id", "names"?}` (an imported state of the graph); `objective` is `"graph-ce"`, `{"kind": "native-policy", "path", "sha256"}` or `{"kind": "native-query", …}`. 200 `{"loss", "stats", "gradient": meta, "native_identity", "objective", "effective_native_identity" \| "invocation_actors", "actor_graph_sha256"}`. (2) `{"native_recorded_objective": {…}}` or a body with `schema: "natlang.native-recorded-objective-request/1"` (`turns`, `events`, `terms`, `targets`, `native_state_modes`?): 200 `{"loss", "stats", "gradients": {root: meta}, "native_identity", "objective"}`. (3) `{"native_derivative_program": {…}}` or a body with `schema: "natlang.native-derivative-program/1"` (`inputs`, `nodes`, `outer`, `wrt`, `arithmetic`): 200 `{"schema": "natlang.native-derivative-execution/1", "program_id", "execution_id", "loss", "arithmetic", "gradients", "outputs", "optimizer_states"}`. Gradients are stored tensor artifacts of kind `parameter-gradient`, `adapter-gradient` or `state-gradient`, held for the requesting owner. Errors: `neuralese-native-grad-schema`, `neuralese-native-grad-order` (a graph request with `order` other than 1, `derived` or `create_graph`: graph requests are first order, and a second-order objective is sent as form 3), `neuralese-native-grad-graph`, `neuralese-native-grad-objective`, `neuralese-native-grad-target`, `neuralese-native-grad-invalid` (any form that fails validation, decision objectives included). |
| `POST /v1/neuralese/optim` | `{"schema": "natlang.native-optimizer-request/1", "native_identity", "gradient_id", "optimizer": {"name": "sgd" \| "adam", "learning_rate"? (1e-3), "weight_decay"? (0), "beta1"? (0.9), "beta2"? (0.999), "eps"? (1e-8), "momentum"? (0)}, "optimizer_state_id"?}`. The live model is not changed. A `parameter-gradient` publishes a new composite checkpoint: 200 `{"checkpoint", "parameters": meta, "optimizer_state_id", "native_identity", "step"}`. An `adapter-gradient` updates the coefficients: 200 `{"adapter": meta, "optimizer_state_id", "step", "native_identity", …}`. A `state-gradient` updates the native state: 200 `{"native_state_id", "native_state": meta, "optimizer_state_id", "update_artifact_id", "step", "native_identity"}`. A server started from an actor view or an inference-only checkpoint refuses. Errors: `neuralese-native-optim-schema`, `neuralese-native-optim-invalid`. |
| `POST /v1/neuralese/adapters` | Create an exact-base actor adapter: `{"parameter_names", "kind"? ("xs" \| "free-lora"), "rank"? (8), "seed"? (0), "alpha"? (free-lora; default `rank`), "basis_id"?, "native_identity"?}` → 201 meta of an `adapter-coefficients` artifact. The adapter's weight delta starts at zero. `xs` also stores an `adapter-basis` artifact (the coefficients' `source_artifact_id`); `basis_id` slices an existing canonical basis. Errors: `neuralese-native-adapter-invalid`. |

Actor adapters change model weights for one request; they are not stored program
guidance. Guidance uses native-state spans, which condition the call's KV and hybrid
state without changing weights.

### Info, health, models, metrics

`GET /v1/neuralese/info`:

| Field | Meaning |
| --- | --- |
| `dialects` | One dialect: `native-state/<architecture>`, or `semantic-state/<adapter_sha256>` with a semantic adapter ([NEURALESE_DIALECTS.md](NEURALESE_DIALECTS.md)). |
| `width`, `dtype` | Hidden width; the embedding weight dtype (`float32`, `bfloat16`). |
| `max_block_length` | The producer-span token budget a view write reserves (64). Chat spans are bounded only by `max_tokens`. |
| `context` | Served context in tokens: `min(max_position_embeddings, -c)`. `view` plans against it. |
| `grad`, `grad_order`, `stream` | `true`, `2`, `true`. |
| `store` | `{"owners": true, "persistent": bool}`. |
| `native_state` | `{"enabled": true, "format": "neuralese.native-state/1" \| "neuralese.semantic-state/1", "semantic_adapter_sha256"?, "consumption_modes", "architecture", "checkpoint_sha256", "tokenizer_fingerprint", "embedding_weight_sha256", "role_marker_ids"}`. `consumption_modes` lists the modes of Consuming that this server can use. |
| `server` | `"reference"`. |
| `rl_policy_binding` | `natlang.rl-policy-binding/2`: the served actor's checkpoint and native identity, history-reasoning policy, Python runtime, and participant, actor view and semantic adapter when present. |
| `prompt_rendering`, `prompt_rendering_identity_sha256` | Serving prompt policy (`full` or `program-header`) and its identity. |
| `capabilities` | The sorted list of capabilities served. |

`GET /health` and `GET /v1/health` answer `{"status": "ok"}`. `GET /v1/models`
answers `{"object": "list", "data": [{"id": "native-<architecture>", "object":
"model"}]}`. `GET /metrics` answers Prometheus text with scheduler counters:
`natlang_native_decode_tokens_total`,
`natlang_native_batched_decode_rounds_total`,
`natlang_native_batched_decode_tokens_total`, `natlang_active_sequences`,
`natlang_native_open_produce_spans`, `natlang_queued_requests`.

### Missing server

A runtime that cannot read `GET /v1/neuralese/info` from its model server fails with
`neuralese-unsupported-backend`; a server that does not list the declared dialect
fails with `neuralese-dialect-mismatch`.

## Native marker failure observations

`native.protocol-recovery` advertises the boolean request field
`x_natlang_native_protocol_recovery` (chat completions and view). Without it, a marker
violation (Producing) fails the request with its code. With it, a malformed draw
answers a completion with `finish_reason: "error"`, the raw sampled text as `content`
(no tool calls, including calls that parsed before the failure), and
`x_natlang_native_failure`:

```json
{ "schema": "natlang.native-marker-failure/1", "code": "neuralese-native-state-…",
  "message": "<code>: <detail>", "trigger": "sample" | "length",
  "sample_index": 0, "fed": false, "own_input_count": 0, "generated_input_count": 0,
  "max_tokens": 0, "stop_token_ids": [] }
```

`code` is one of `neuralese-native-state-open`, `-close`, `-consume-marker`,
`-nested`. `trigger` is `length` (with `sample_index: null`) only for a span left open
at `max_tokens`; otherwise `sample_index` is the offending final draw and `fed` says
whether it was fed. The matching `x_natlang_rl_capture` uses
`natlang.native-rl-sampling/4` at positive temperature or
`natlang.native-structural-capture/2` at temperature 0, with the descriptor as
`protocol_failure`. Capture retains the actual actor, normalized draw log-probabilities,
forced inputs, producing flags, imports and completed writes. Greedy observations are
not policy-gradient samples. An unfinished span produces no state. `usage.completion_tokens`
counts the sampled draws and forced inputs.

Clients preserve the failed turn before feeding its typed rejection back to the same
invocation. All sampled actions and complete episode occurrences remain part of graph
replay. A native view failure contains every `source_turns` entry through the failed
write and the failure, with no final `id`. An owner holds the states its request
produced, including those produced before the failure; collection by another owner
cannot remove them.

## Agreement

Cached execution must agree with explicit execution on the same committed sequence.
The shared-state tests check that incremental capture and splicing match one-pass
graph execution in outputs and gradients
(`tests/neuralese/test_shared_state.py`), and that batched decoding of unequal cached
sequences, producing and not, matches each sequence decoded alone in states, outputs
and gradients (`tests/neuralese/test_native_decode_batch.py`). Guidance rollback
restores the cache from the prompt anchor and replays the retained inputs with their
recorded roles.
