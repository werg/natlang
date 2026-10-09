# S4: Runtime and inference servers

Draft, 2026-10-03. Detailed plan for stage S4 of the [Neuralese programme](README.md). It implements the language and protocol of [S0](S0_SPEC.md) in the native runtime and makes Neuralese a first-class target of real inference servers. It uses the model modules and procedures of [S3](S3_PORT.md) and hosts the execution environments listed in [S1](S1_DATA.md) §5.3.

S4 starts in parallel with S3. Until G1, every server runs an **untrained port**: written blocks are the input embeddings of a short text the model emits inside the literal, and reads splice stored vectors as usual. This exercises every path end to end before trained weights exist. After G1, the trained `nd:natlang@1` weights replace the stand-in without interface changes.

## 1. Starting point

| Area | Today | Consequence |
| --- | --- | --- |
| Model contract | `ts-host/src/contracts.ts`: `ModelTurnRequest = { messages, tools, temperature, seed, max_tokens, tool_choice }`, `ModelTurn = { calls, text, … }`. Text only. | The contract gains Neuralese content parts (§3). |
| Transports | `ts-host/src/model/chat-completion.ts` (OpenAI-style chat JSON), `openai-compatible.ts`, `local-server.ts`, `llama-runtime.ts` (managed llama.cpp server releases), the browser path through wllama. | Every transport learns the new parts, or rejects them. |
| Types and values | `native/types.ts` (`prim`, `lit`, `record`, `list`, `dict`, `union`, `name`, `lambda`, `host`), `native/values.ts` (`Value`, live objects, `CaptureCell` with a `mutable` flag, `coerce`). | A `neuralese` kind and a reference value type are added. |
| Compiler | `ts-host/src/compiler/` lowers `nl`, `iterateOn` and eval code through a TypeScript transformer (`lower.ts`, `inline.ts`, `nl-call.ts`, `intrinsics.ts`, `policy.ts`, `eval-check.ts`). | Literals, context binding, termination checks and the rewrite pass live here. |
| Trace | `native/trace.ts`, `reduction-trace/1`. | Extended to the graph record. |
| Iteration | `runtime/iterate.ts` requires a measure or step limit for every iteration and runs natural-language predicates without a dedicated prompt. | Changed to the S0 §8 and §13 rules. |
| Python serving | `scripts/serve_improvement_student.py`: a 71-line HF Transformers + PEFT HTTP server. `scripts/serve.sh`: llama.cpp's official CUDA image with GGUF models. | The reference server is new code in `training/neuralese/serve/`; the old script stays for crisp serving. |
| llama.cpp source | `vendor/llama.cpp` is a sparse checkout at upstream `972d2313` holding only `convert_hf_to_gguf.py`, `gguf-py` and conversion helpers; `gguf-py` already knows the LFM2 architecture. Serving uses the official image. | The full source must be fetched and forked (§6). Upstream `llama_batch` accepts embeddings (`embd`), and LFM2 is an upstream architecture. |
| vLLM | No source checkout. `~/vllm/.venv` has vLLM 0.15.1, which supports `enable_prompt_embeds`. The DGX runs the `vllm-node` image. | Reads can use prompt embeddings; the write procedure needs a model-runner extension (§7). |

## 2. Runtime changes

All paths are under `ts-host/src/` unless stated. Each item cites the S0 section it implements.

### 2.1 Types and values (S0 §2)

- Add `{ kind: 'neuralese'; element: Type; dialect: string }` to `native/types.ts`, parsed from `Neuralese<T>` and `Neuralese<T, D>` in frontmatter, inline signatures, eval-declared types and `.nz` headers. `DefaultDialect` resolves from program configuration.
- Reject `Neuralese<Neuralese<T>>` (`neuralese-nested`) and recursive function types (`type-recursive-function`) in the type environment, including through aliases.
- Add the reference value `{ "$neuralese": { type, id } }` and the soft-function value `{ "$neuralese-fn": { body, captures } }` to `native/values.ts`. `coerce` checks the element type and dialect and never inspects the payload.
- Applicability: a value of type `Neuralese<F>` with function `F` becomes a callable host object whose call starts a soft-body invocation (§2.6).

### 2.2 Opacity checks (S0 §2)

The TypeScript surface (`spec/neuralese.d.ts`, branded `SoftValue`) gives most opacity errors for free in application code and callable-folder TypeScript. `compiler/eval-check.ts` adds the natlang diagnostics for eval code: `neuralese-opaque-access`, `neuralese-condition`, `neuralese-interpolation`, and rejects serialising a payload. The runtime repeats the checks on values that reach host code as data.

### 2.3 Literals in both directions (S0 §3)

- **Parsing a model turn.** A transport returns text and tool-call arguments with Neuralese parts in place (§3). The runtime stores each block (or receives its ID from the server), and replaces the literal in eval code with a reference expression before compilation, so the checker sees ordinary TypeScript. `neuralese-untyped-literal` fires when no contextual type exists.
- **Rendering a conversation.** Wherever the opening declarations, eval results, staged results, soft bodies or context items show a Neuralese value, `native/agent.ts` emits a Neuralese content part instead of a preview. Cut-offs never split a block; a block that does not fit is cut as a whole with the usual note naming where the value is held.
- **Escaping.** Content text is sent with special-token splitting off; only the parts produce control tokens. A conformance case sends `<|neuralese|>` as file content and checks that it stays text.

### 2.4 Tensor-store client (S0 §3.4)

`native/neuralese-store.ts`: content IDs (`nz1_` + base32 SHA-256 over dialect, shape, dtype, bytes), metadata records, put/get/pin against the server endpoints, a local cache keyed by ID, and garbage collection of unreferenced entries. The runtime never holds float arrays in its value model; it holds references and moves bytes only between store, files and servers.

### 2.5 Combinator library (S0 §4)

`natlang:neuralese` is a surface module. `map`, `zip`, `ap`, `combine`, `split`, `splitList`, `read`, `convert` and `gloss` are system natural-language functions whose soft bodies are exports of the standard library's own `.nz` file, loaded like any context. `empty` is a runtime primitive (a zero-length entry). Before trained bodies exist, each operator's body is a text-initialised block, so the library works with the untrained port. `read` validates through the existing completion path and fails with `NatlangCallError`.

### 2.6 Neuralese functions and captures (S0 §6)

- `nl.with({ … })` with a literal body creates a soft-function value. Captures are snapshots by default; `live(x)` creates a live `CaptureCell` with version-checked write-back, and marks the function as unsavable.
- Function-typed bindings are always captured by value, in text `nl` and in literals. `compiler/inline.ts` enforces this when it resolves implicit captures; text `nl` otherwise keeps implicit live captures and `let` write-back unchanged.
- Executing a soft function renders the call like a text call with the body literal in the instruction section and the captures listed in the opening scope.

### 2.7 Contexts (S0 §7)

- `native/context.ts` (new): a context is an immutable, content-addressed folder value (a Merkle tree over `.nl`, TypeScript, `.nz` and subfolder entries). Folder overlays and continuations reuse the same storage.
- Binding: the compiler binds each function to its definition-site context (the companion-folder and `natlang.d/` rules). `foo.in(context)` rebinds.
- **Context interface.** The compiler computes each function's free names with types and stores them in its compiled metadata. Rebinding checks the new context structurally against it.
- **Executable nodes from files.** A context records, for every executable node, the loaded file context it came from. Rebinding rejects a node absent from every loaded file context; edited nodes are recompiled against their interface; data entries are free.
- Directory reducers over contexts return new context revisions; `folder.apply` stays the commit for real folders.
- The identity-based caller-chain guard is retired for natlang definitions; the runtime check stays only for host TypeScript callbacks into natlang.

### 2.8 `.nz` files (S0 §5)

- `native/nz.ts`: read and write the safetensors container (header in metadata, tensors keyed by content ID); validate each export against its declared type on load; resolve captures between exports and across files.
- Declaration generation: the build emits `foo.d.nz.ts` from the header so imports type-check, as `.nl` files get `foo.d.nl.ts`.
- `natlang nz show file.nz` prints the header with types and references.
- `save(path, exports)` from `natlang:learning` writes a file through the same module.

### 2.9 Graph-record trace (S0 §11)

Extend `native/trace.ts` to the graph record: block-write nodes (context reference, stop decisions, length, truncation, entry), block-read nodes, readouts, combinator calls, applied rewrites, context revisions, effects with recorded results, `grad` calls and iteration steps, with producer/consumer edges. A new trace version is registered; old traces stay readable.

### 2.10 Compiler rewrite pass (S0 §4.3)

A pass in `compiler/` after type checking matches the six rules (map fusion, read/map commutation, map identity, combine reassociation, combine identity, split of zip). A rule fires only once it has been enabled for the current model and dialect version by a whole-program comparison showing no harm (S0 §4.3); the comparison reruns when the model or version changes. Fusion inserts the system `compose` operator. Each applied rewrite is traced; rewrites can be disabled per program or call site.

### 2.11 Iteration (S0 §8, §13)

In `runtime/iterate.ts`:

- Require `withLimit` or `withMeasure` only when the `until` predicate is TypeScript, or when `checkProgress('off')` is set.
- Run natural-language predicates under the dedicated stopping-condition system prompt (iteration count, whether recent states changed, bias toward accepting a reasonably met criterion).
- Keep statistics-scheduled progress reviews; update the built-in help and `spec/SPEC.md` text together.

### 2.12 Learning service (S0 §9)

`natlang:learning` exposes `grad`, `valueAndGrad`, `stopGradient`, the objectives, the optimisers and `save`, only to callers given the service. The runtime implements them as requests to the server's replay sessions (§4.4): it sends the graph record of the loss evaluation and the argument references, and receives a loss and opaque gradient and optimiser-state references. Nested `grad` defaults to first order. `logLikelihood` sends a recorded trajectory.

## 3. Model-turn protocol and block endpoints

- **Content parts.** Message content and tool-call arguments may contain `{ "type": "neuralese", "id": "nz1_…" }` parts, in requests and in responses. In a response, a part marks a block the server wrote; its entry is already in the server store.
- **Tool-call arguments.** LFM2.5 uses Pythonic tool calls in `<|tool_call_start|>[…]<|tool_call_end|>`. A literal inside a string argument (for example `eval(code="const p: Neuralese<Plan> = <block>;")`) is returned as a structured argument: an array of text and Neuralese parts. `chat-completion.ts` reassembles it, and the runtime treats it as source text with an embedded literal.
- **Endpoints.** `PUT /v1/neuralese/blocks/{id}`, `GET /v1/neuralese/blocks/{id}`, `POST /v1/neuralese/blocks/{id}/pin`, and `GET /v1/neuralese/dialects` (dialect versions spoken and the rewrite rules enabled for them).
- **Learning sessions.** `POST /v1/neuralese/grad` (graph record, argument references, objective, order) returns loss, gradient and optimiser-state references; `POST /v1/neuralese/optimize` applies an optimiser step server-side.
- **No fallback.** A transport whose backend lacks a dialect fails with `neuralese-unsupported-backend`. Capability is probed once per runtime through `/v1/neuralese/dialects`.
- **Collector.** `ts-host/scripts/teacher-collector.mjs` and the materializers record Neuralese parts as references; teacher models without Neuralese cannot run Neuralese programs, which is the intended failure.

## 4. Python reference server

`training/neuralese/serve/` in the S3 package, using its model wrapper, heads and procedures.

### 4.1 Tensor store

Content-addressed entries in memory with a disk spill directory, pinning, and the S0 entry metadata. The same store module backs training replay.

### 4.2 Generation

A continuous-batching decode loop over HF `Lfm2ForCausalLM` with the S3 port modules:

- Ordinary decoding with the native chat template and tool-call parsing.
- When the LM head emits `<|neuralese|>`, that sequence enters the write procedure (S3 §3.1): shallow recurrence for that sequence while others continue text decoding, then blockwise completion, then readback. The store entry is created and a Neuralese part is emitted.
- Reads splice stored vectors through `A_in` and the interface norm (S3 §3.2).
- The runtime hard maximum per block is a server setting; reaching it sets `truncated`.

### 4.3 Untrained stand-in

Before G1: no shallow writer exists, so a written block is the full model's text inside the literal, converted to its token embeddings and stored as a block. Reads are real splices. This tests protocol, runtime, store, files, contexts and conformance plumbing; it does not test writing quality.

### 4.4 Replay and gradient sessions

`grad` requests replay a recorded graph on the GPU (S0 §11.3): restore context revisions, supply recorded effect results, hold discrete choices fixed, recompute the differentiable path, and accumulate gradients at the requested argument entries. Batching across cases happens inside one replay. Gradients and optimiser states are store entries. No gradient checkpointing.

### 4.5 Execution environments

The host services built for S1 (SQLite, unit-test runners, world simulators, SWE containers) run beside the runtime, not inside the model server. They are registered as natlang services with declarations, deterministic fixtures and traced effects.

## 5. Conformance and parity

- **Suite.** `conformance/neuralese/` (written in S0) plus the S3 evaluation-set checks for `nd:natlang@1` (S3 §8): literal round-trips, escaping, combinator types and failures, `.nz` loading, context rebinding, termination rules, iteration rules, rewrite enabling, graph replay, readback agreement.
- **Parity against the PyTorch reference.** For a fixed set of stored blocks and prompts, each server must match the reference on: read-port logits after splicing (within tolerance), greedy continuations after a block (≥ 99% identical, as in G1), written block length and stop decisions under greedy stopping, and payload vectors within tolerance after dequantisation. Quantised builds report their deviation and must still pass the dialect's consumer thresholds.
- **Runtime parity.** Node and browser runtimes produce the same traces for the same programs on the same server, as for crisp natlang today.

## 6. llama.cpp extension

Work in a full fork of upstream llama.cpp, vendored at a pinned revision beside the existing sparse checkout, with patches kept small for upstreaming.

- **GGUF.** Extend `convert_hf_to_gguf.py` and `gguf-py` for the S3 modules on LFM2: control-token rows, feedback projection (including the mixture branch and its temporary readout if retained), stop head with its position embedding, content projection, interface norm, adapters where a model needs them, and metadata for the cutoff `k`, the dialect ID and version, and `L_max`.
- **Reads.** Build batches that mix token IDs and embeddings; positions are contiguous. Upstream `llama_batch.embd` supports embedding input; mixed batches may need sequential sub-batches per sequence.
- **Write graph.** For LFM2: a shallow graph running layers `0…k−1` for one position per step with the feedback projection and stop head, using the shallow layers' KV and convolution caches; a blockwise graph for layers `k…D−1` over the collected residuals from the upper caches at the block start; cache snapshot and restore for KV positions and convolution rolling windows; readback prefill.
- **API.** A `llama_neuralese_*` C API (begin block, step, complete, read back, get payload), and the server's chat endpoint emitting and accepting Neuralese parts and the block endpoints of §3, with a store in the server process.
- **Browser.** Build wllama from the fork so the browser runtime gets the same capability; blocks are stored in the Origin Private File System (OPFS) and move between it and `.nz` files by ID.
- **Managed runtime.** `model/llama-runtime.ts` pins Neuralese-capable releases of the fork; `scripts/serve.sh` gains a Neuralese image.

## 7. vLLM extension

- **Reads.** Use `enable_prompt_embeds` for prompt-side splicing; verify the version in the DGX image and in `~/vllm/.venv` (0.15.1). Mid-sequence reads after generation require the request to be re-prefilled as embeddings or a model-runner hook.
- **Write procedure.** A plugin model class for LFM2 with the port modules, plus a model-runner hook that switches a sequence into shallow-step mode when it emits `<|neuralese|>`, runs the blockwise completion, restores its cache, and prefills the payload. Other sequences in the batch continue normally.
- **Store and endpoints.** The block endpoints of §3 in front of the engine, with a shared store.
- **Purpose.** High-throughput rollouts for S5 and S7 on the DGX. Parity against the reference as in §5.

## 8. Migration

- **API migration entries** in `training/api-migrations/` for: Neuralese content parts in model turns; the context-as-argument revision (callable folders, directory reducers, retired caller-chain guard); function-typed captures by value; iteration limit and predicate-prompt changes; the trace version.
- **Data.** Existing corpora are rebuilt through the S1 migration (eager typing, explicit captures) under the new runtime version. Data built under different runtime versions is not mixed.
- **Programs and examples.** Existing programs keep their meaning; examples that use unbounded TypeScript-predicate iteration gain explicit limits; examples that capture function-typed `let` bindings are checked for changed behaviour.
- **Adaptation artifacts.** `natlang.adaptation/v1` gains context revisions and `.nz` references for context-held trainable values.

## 9. Work items

Ordered by dependency; items without a dependency on S3 weights proceed before G1.

1. Fetch and vendor a full llama.cpp fork at a pinned revision; record vLLM image versions on the DGX.
2. Type kind, reference values, opacity diagnostics, recursive-function-type check (§2.1–2.2).
3. Tensor-store client and the S3-package store; block endpoints in the Python server (§2.4, §3, §4.1).
4. Protocol: content parts in contracts, transports and collector; tool-argument parts (§3).
5. Literal parsing and rendering in the runtime (§2.3).
6. Python server generation loop with the untrained stand-in (§4.2–4.3).
7. Contexts, rebinding, interface checks, executable-node provenance; retire the caller-chain guard (§2.7).
8. Function-typed captures by value; soft-function values and `live` (§2.6).
9. `.nz` reader/writer, declaration generation, `nz show` (§2.8).
10. Combinator library with text-initialised bodies (§2.5).
11. Iteration changes and predicate prompt (§2.11).
12. Graph-record trace (§2.9) and replay sessions with `grad` (§4.4, §2.12).
13. Compiler rewrite pass with gating (§2.10).
14. Execution environments as services (§4.5), with S1.
15. llama.cpp: GGUF conversion, reads, write graph, API, server endpoints (§6).
16. vLLM: reads, write hook, endpoints (§7).
17. wllama build and browser parity (§6).
18. Swap in trained `nd:natlang@1` weights after G1; run the full conformance and parity suites on every server.
19. API migration entries and example updates (§8).

## 10. Exit criteria

- With trained S3 weights, every S0 conformance case passes on the Python reference server and on llama.cpp, and the read and write parity checks of §5 pass on llama.cpp, vLLM and the browser build.
- A Natlang program that writes, stores, imports, rebinds, combines and reads Neuralese values runs identically on Node and in the browser.
- `grad` through a context item and an `iterateOn` training loop with `adam` run end to end and save a `.nz` file.
- Every applied compiler rewrite appears in the trace with its gating measurement.
- A backend without Neuralese support fails such programs with `neuralese-unsupported-backend`.
- The API migration entries exist and the migrated corpus renders and replays.

## 11. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Upstreaming.** The llama.cpp and vLLM changes are developed in forks, designed for upstreaming: general APIs (mixed token/embedding batches, a block-write procedure, model-declared heads and adapters), small patches, upstream conventions.
2. **Tool-call serialization.** Literals stay inside the eval code string of the native Pythonic tool call, with the control tokens inside the quoted string. Tool-call parsers recognise the control tokens there; no dedicated argument form.
3. **Browser store.** Browser runtimes store blocks and `.nz` files in the Origin Private File System (OPFS), keyed by ID.

## Current raw/latent GGUF support boundary (2026-10-09)

The fork now transports the optional `full-residual-v1` read adapter on the
legacy `legacy-rms-v1` projector. The adapter runs inside `nz_read` after the
legacy interface norm; ordinary text encoding does not use it. CPU parity used
a nonzero adapter. This does not qualify the new foundation runtime.

The exporter still refuses `raw-token-v1` and `latent-sketch-v1/v2`. Their
projector tensors and write/stop semantics differ from the legacy RMS/vocabulary
mixture graph, so removing the refusal would not be a valid port. The required
profile work, LFM2/Mellum coverage limits and acceptance sequence are in
[`READ_ADAPTER_PORT_HANDOFF.md`](READ_ADAPTER_PORT_HANDOFF.md). The exact build,
source pins and CPU parity summary are in
[`runs/neuralese-read-adapter-cpp-port-20261009-v1/verification-receipt-v1.json`](../../runs/neuralese-read-adapter-cpp-port-20261009-v1/verification-receipt-v1.json).
