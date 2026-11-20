# Engine parity

natlang serves models through four engines. A serving improvement (a request field, a cache, scoring, sampling,
capture) lands on every engine it applies to in the same push. Where it cannot, it gets a row here with the reason and
the next action, and the row leaves when the gap closes.

| Engine | Code | Who runs it | How it stays current |
|---|---|---|---|
| Managed llama-server (the fork) | werg/llama.cpp-neuralese, pinned in `ts-host/src/llama-runtime-release.ts` and `llama-fork-source.ts` | `natlang` CLI, library hosts | `node scripts/update_llama_runtime.mjs --fork natlang-bNNNN` |
| In-page engine (wllama) | werg/wllama (pinned in the fork's `.github/natlang-wllama.json`) compiled with the fork's tree, vendored in `ts-host/vendor/wllama{,-compat}` | browser runtime, website | the same command, from the same release (its `llama-natlang-wllama.tar.gz`); `ts-host/test/llama-runtime.test.mjs` fails when the vendored build's fork commit differs from the runtime pin |
| Python native engine | `training/neuralese/natlang_neuralese/serve/` | training, exact reference, gradients | one repository with the client |
| SGLang actor | `serve/sglang_actor.py` in front of werg/sglang-natlang | RL collection, held eval | one repository with the client; experiments adopt it at their next pin |

Mechanical guards:

- The client builds score items and reads score results in one place (`ts-host/src/model/decision-scoring.ts`), for the
  managed server and the page alike.
- `tests/neuralese/test_engine_parity.py` fails when the client sends a score-item field the Python servers have not
  declared in `native_learning.SCORE_ITEM_FIELDS`.
- The fork's release workflow builds the in-page engine in the same run as the native archives, so a fork change that
  breaks the WebAssembly build fails the release.
- `scripts/update_llama_runtime.mjs --fork` refuses to move the pin past a fork-own commit (one without an upstream
  `(#NNNN)` suffix) whose short SHA is not in this file: each fork change is reviewed against the other engines, as a
  gap row or in the list below. It also vendors the page build from the same release, and
  `ts-host/test/llama-runtime.test.mjs` fails when the vendored build and the runtime pin come from different releases.
- `node scripts/wllama_smoke.mjs --model M --server S` compares the page's decision scores with the native server's
  (ranking and a 1-nat tolerance, for the items and for a repeat after a batch); run it on every pin bump. natlang-b12074:
  same ranking on 3 items, largest gap per item 0.58-0.67 nats from native, a repeat within 0.16 nats. With a natlang GGUF it
  also checks native states both ways, and that both engines render and tokenize every item alike (prompt and reply
  token counts). natlang-b12081 on natlang-lfm3-5ff07638 (Q8_0, CPU, against the release's linux-arm64 server): passes;
  the server's state put into the page is consumed exactly and a decision on it ranks like the server's (largest gap 0.27
  nats); the page's grammar-forced state has the server's text, tokens and identity, reads back as stored, and the
  server accepts it and consumes it exactly. Its default tolerance is 2.5 nats (next section).

## Numerics: a score is reproducible only under a fixed cache layout

**Q8_0 engines differ by up to ~2 nats per decision score, and that is the int8 rounding of the activations.** On the
smoke items (LFM3-1.7B Q8_0, 6-7 reply tokens, scores -50 to -71 nats), six configurations (the page; a local
`-mcpu=native` build; the b12080 release with flash attention auto, off, off with F32 KV, and BF16 KV) deviate from the
same weights dequantized to F32 (`llama-quantize --allow-requantize ... F32`) by 0.36-0.76 nats RMS per score, 0.58
pooled, and pairs differ by 0.51-1.75 nats at most over the 9 scores. With the F32 copy, flash attention and the KV type
move a score by at most 0.03 nats, and thread count changes nothing. The weights are bit-identical between the two
models, so what remains is the Q8_0 matmul quantizing its activations to int8 blocks: any upstream change (kernels, flash
attention, KV type, WebAssembly vs NEON) redraws that rounding error. The scatter predicts the gaps (difference sd 0.82,
expected largest of 9 about 1.7 nats), and it predicts the direction: the rounding noise flattens the distribution, so
improbable replies score higher than in F32 (mean +0.27 to +0.61 nats in five of the six). The page is not an outlier
(RMS 0.53, the third closest to F32). The b12074 smoke's 0.58-0.67 and this one's 1.11 are two draws of the same scatter.
It is not reducible without unquantized activations (an F16/BF16 GGUF); the Python native engine and SGLang serve BF16
weights and do not have it. `wllama_smoke.mjs` therefore allows 2.5 nats, three sd of the difference.

A score is bit-reproducible when the sequence's KV cells sit in the same order every time. The managed server runs one KV
stream per slot (no `--kv-unified`), so a prompt always fills its stream from cell 0 in position order and a repeat is
bit-identical. The page runs wllama's unified KV cache (all slots share one context), so a sequence's cells land wherever
earlier requests left room. The attention sums run in cell order, and the order changes the last bits; the int8 rounding of
the activations in the Q8 matmuls usually absorbs that, but a value that sits on a rounding boundary flips by one int8 step
and the change grows through the later layers. Measured on LFM3 Q8 (fork dc9410950, 2026-11-14): the attention output of a
layer differs by at most 6e-8 between the two layouts, with the same softmax weights; most option tokens score
bit-identically, a few by 0.001-0.14 nats, an option sum by up to 0.5 nats. The idle-slot prompt cache, decision
checkpoints and the scheduler are exact: identical serialized state before every decode. The smoke test therefore compares
page repeats by ranking and tolerance. Bit-exact page repeats would need one stream per slot, which divides the page's
context between its slots.

## Fork commits reviewed for parity

| Commit | Change | Other engines |
|---|---|---|
| a0c92ce36, 0da69f720, ff6e734e3, e51ca6d68 | decision scoring from a shared-prefix checkpoint, score batching with generating slots, `cache_messages` | rows above |
| a6401bbd3 | streamed reasoning of a guided reply | every engine: the page from natlang-b12074; Python native and SGLang from 2026-11-14 (`serve/native_guidance.py`: guidance blanks the open reasoning, it streams as `reasoning_content`, the rest is held back until final). Unguided streams on both Python engines send open reasoning as `reasoning_content` too, the closing tag dropped (`serve/http.py`, 2026-11-14), as the managed server streams it |
| 26a65db62 | the release builds the in-page engine; deepseek41 compiles on wasm32 | page only |
| dc9410950 | RL capture under top-p | row above; native and SGLang already did it |
| 6d270f086 | top-p RL capture draws on the host (dc9410950's device default overrode it); omitted top_p is no nucleus | row above |
| cab27bfbe | a context checkpoint that does not load falls back to re-processing the prompt instead of aborting; the in-page build gains slot save/restore (werg/wllama 0095d0f) | page warm openings row; RL capture under top-p (dc9410950, 6d270f086) now released on every engine |
| 69cba25d9 | native-state put after startup is one main-loop task (`SERVER_TASK_TYPE_NATIVE_STATE_REGISTER` with the bytes: verify, store, register) for the HTTP PUT and embedders; `server_context::get_native_state` | page: werg/wllama 5b0dcd3 `putNativeState` / `getNativeState`; section "Native state" |
| 623fd10c8 | the release builds werg/wllama 5b0dcd3 | page only |
| cb03daee7 | a streamed reply's last chunk carries its produced native states (`native_state`, and `x_natlang_message` with each state as a part); before, a stream carried only the placeholder text and lost which state it named | page (it streams; found by the playground native-state smoke) and the managed server's streamed turns; the Python native engine streams parts already; SGLang does not produce in-reply states |
| 401f8ad5e | the RL actor binding (GGUF and executable hashes) is computed on first use, not at load: a natlang GGUF failed to load in the page (`cannot hash actor executable`), and every load read the whole model file to hash it | managed server: `/v1/neuralese/info` and RL captures unchanged (`check_llama_native_state_protocol.py --expect-rl` passes); Python native and SGLang bind their actors separately |
| b704a8b2f | native-state files: `--native-state` and the page's `native_states` load natlang's store files (a release's `states/<id>.nzs`) through the PUT path (verified once, stored as read); PUT `{"path"}`; non-finite values refused; GGUF state-span format removed | every engine: section "Native state: one format, handed over as files" |

## Native state: one format, handed over as files

Every engine and store holds a native state as natlang's block-store file (`serve/store.py`: safetensors with the
`natlang.native_state` header and the content ID), keeps the bytes as received, verifies them once on entry (layout,
finite values, content ID in one pass) and decodes a state only where it computes on it.

| Engine or store | Produces | Imports | Moves without decoding |
|---|---|---|---|
| Managed llama-server (fork cb03daee7, natlang-b12081) | produce spans, encoded once into the store file | PUT body or `{"path": FILE}`, `--native-state FILE` (a release's `states/<id>.nzs`); every put after startup is one main-loop task (verify, store, register); one decode for span registration | GET answers the stored bytes |
| In-page engine (natlang-b12081 wllama build, werg/wllama 5b0dcd3, vendored) | produce spans of the page's default model, natlang's own GGUF (natlang-lfm3-5ff07638: producer adapter and native identity), through the server's code; `BrowserLocalModel` archives each produced state into the page's store after the turn (`getNativeState`, the bytes as stored) | `native_states` Blobs at load (a release's `states/<id>.nzs`), and after load `putNativeState` (the server's put task): a request naming a state the engine lacks gets it from the page's store (`restoreNativeStates`, as with a server) | the wire moves between the OPFS store and the engine as bytes; only the engine decodes |
| Python native engine | produce spans (`put_native_state`, `write_native_state_file`) | splices `load()`ed blocks (one copy from a file mapping) | HTTP GET `sendfile`s the stored file, PUT streams to a file and adopts it, metadata from the header (`NativeStateFile`) |
| SGLang actor (fork b6b2319c7) | the fork writes the store file; the store adopts it by rename | the fork reads the stored file by path (`layers.*` only) | the front never decodes |
| Python block store / learner `NativeStates` | | `read_native_state_file` + `load()` | adopt by rename, verify from a mapping |
| ts-host stores (memory, file, OPFS) and the collector archive | | | `putNativeStateWire`/`getNativeStateWire` keep the wire as received; decoded tensors are views of it; files re-verified when read from disk; archive and restore move wires between stores |

**Soft instruction and skill states (tuned releases' `softStates`) are native states, so every engine serves them.**
A request carries each as a `{"type": "native_state", "id"}` part where its text stood, and several states in one request
compose by concatenation (CSPD's composition; docs/TUNING.md). Per request in one batch: the managed fork (per-sequence,
`--native-state` files), the Python native engine and the SGLang actor (`natlang_import` per request,
`sglang_actor.py` `_sglang_import`); the page serializes requests. Since 2026-11-20 `natlang tune` trains them on the
managed backend by default (`tune.softStates: false` opts out; `tune/pipeline.ts` `softStatesServed`). No gap.

**The learned operators' in-context capture** (OPERATORS_RESTORE_20261117.md) has a managed HTTP implementation and an engine primitive shared by the fork and page; the page runtime binding remains blocked as recorded below. The Python engine and the SGLang actor serve `native.capture`: the actor goes through the fork's capture request, the request its forced-producing parity covers. The managed llama-server and the page serve no write endpoints and write states only from produce spans; the managed HTTP `grammarCapture` client sends the call's own messages again with a grammar of the exact ids `<reply prefix> produce_open <content> produce_close`.

**A capture whose prompt imported state** (an operator call under its own soft prompt state, 2026-11-20): the call's RL
capture records the import as a `native_state` prompt segment, and the capture keeps it in place on every engine. The
Python engine and the SGLang actor take `prompt_segments` in `native.capture` (the SGLang front sends the fork one
capture request with the state's rows as `natlang_import`; `test_sglang_actor.py` checks it against the Python engine's
capture tensor by tensor). The fork and the page re-send the call's own messages (the state as a `native_state` part) with
the forced span (`grammarCapture`). Each returns the source capture with the state as a `native_state` segment of
`token_count` rows. The learners read it as fixed stored context inside the producer document
(`lower_rollout_turns.write_sources`, `lfm3_fast_train.passthrough_plan`). No gap.

Live check, 2026-11-17: natlang-b12081 CPU with the natlang-lfm3-5ff07638 Q8_0 GGUF against the Python engine (BF16, CPU) on 5ff07638, two view sites, 400-token spans.
- The fork's rendered prompt equals its RL capture's prompt ids.
- The two states hold the same token ids and the same 38 tensors.
- Relative L2 difference: median 0.025-0.026 and largest 0.045-0.048 (layer-12 V), the Q8 floor. For comparison, SGLang vs Python forced producing differs by 0.050 (fork-checks).

## Open gaps

| Feature | Fork | Page | Python native | SGLang | Next action |
|---|---|---|---|---|---|
| Decision: reuse the function's opening across inputs (`cache_messages`, fork e51ca6d68) | yes | yes from natlang-b12074 | equivalent: the prefix cache resumes from any stored grid block | yes, to within 255 tokens (2026-11-14, `sglang_actor._sglang_warm_opening`, on by default, `NATLANG_SGLANG_WARM_OPENINGS=0` opts out): with `extra_buffer` the fork (ace8d6868) snapshots conv state once per prefill, at `prefix + floor(extend / 256) * 256` (`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`; grid lcm(LFM3 `mamba_chunk_size` 256, page size)), and a finished request inserts its tokens only up to that snapshot (`MambaComponent.prepare_for_caching_req`); a request resumes only from a state on its own path, so each input's state lies inside its own input, the second input re-prefills the opening (only it tracks a branching state there, `finalize_match_result_in_tree_core`), and concurrent items of a batch all do. The actor therefore renders the first `cache_messages` messages, checks they are a token prefix of the prompt, and sends one prefill request of exactly those tokens per distinct opening (concurrent items wait for it; re-sent after 60 s or a weight update) before any option is scored: each input then prefills at most 255 opening tokens plus its own. Exact from the next pin, fork 33230160b (`training/neuralese/sglang-fork.json`): the warm request sets `custom_params.natlang_track_input_end`, the fork's scheduler cuts its prefill so the last extend is a whole number of grids and the state lands on the boundary (`PrefillAdder._natlang_input_end_extend`), and every input prefills only its own messages; the front detects the fork's `NATLANG_TRACK_INPUT_END` marker and keeps the grid-aligned warm against older pins. An opening shorter than 256 tokens stores no state on either | at the next pin on the DGX, read `natlang_sglang_cached_prompt_tokens_total` / `natlang_sglang_prompt_tokens_total` (and `natlang_sglang_opening_warms_total`) over decision calls of one function with different inputs (on fork 33230160b, cached tokens should reach the full opening) |
| natlang's GGUF as the public playground's model (native states on the public page) | n/a | the local page: yes (`model-default.ts`, natlang-lfm3-5ff07638); the public page: no, it keeps the stock LFM3 Q4_K_M from Hugging Face (`build-playground-site.mjs`) | n/a | n/a | blocked on a host: the GGUF is a GitHub release asset, and neither `github.com/.../releases/download` nor its redirect target sends `Access-Control-Allow-Origin`, so a page on another origin cannot fetch it (the site smoke's `--network` check failed, 2026-11-16). A CORS-capable host (e.g. a Hugging Face model repo, as the stock model uses) is the owner's call; then point `build-playground-site.mjs` at `BROWSER_MODEL_RELEASE` |
| Training guidance turn (`x_natlang_privileged_context`, `x_natlang_guidance_policy`; capability `training-guidance`) | no: refused by ts-host (`training-guidance-unavailable`, `model/neuralese-server.ts`), never dropped | no: refused (`browser/local-model.ts`) | yes: renders the runtime's guidance message (`natlang.guidance-request/2`, ts-host `guidanceMessage`) with its template after the leading system message and marks its span (`engine.py` `_guidance`, rl/privileged_context `message_span`) | yes (inherits `_guidance`, `sglang_actor.py:619-621`) | The guidance turn has one definition, the runtime's system message (2026-11-18). Blocked on the fork's capture: it marks privileged tokens only from a pre-tokenized `x_natlang_privileged_block {insert_at, token_ids}` (`tools/server/server-rl-capture.cpp:43-54`), cannot place a writer state in it, and returns no guidance receipt. Next: a fork release (`update_llama_runtime.mjs --fork`) that renders the request's guidance message after the leading system message, flags its span and echoes the receipt, as `message_span` does; the page also needs that release. |
| Decision: score items of one batch concurrently | yes | yes (one engine request) | serial (one execution lock) | yes (2026-11-14) | none |
| Decision: one prefill shared by all options | yes (parent slot checkpoint) | yes | yes | yes: the first option warms the radix cache, the rest run in parallel (each option once since 2026-11-14) | none |
| Score turns with generating slots (fork 0da69f720, ff6e734e3) | yes | n/a (one request at a time) | n/a | SGLang's scheduler; unmeasured | measure decision latency under collection load |
| Prefix-cache hit telemetry | upstream counters | n/a | `natlang_prefix_cache_hits/misses` | `natlang_sglang_prompt_tokens_total` and `natlang_sglang_cached_prompt_tokens_total` (2026-11-14) | none |
| Warm openings kept across restarts | yes (`--slot-save-path`) | yes from natlang-b12076 (`browser/warm-openings.ts`, on by default for a model loaded with its `sha256`; werg/wllama 0095d0f, `capabilities.slotState`): the opening is prefilled once, its slot state saved to OPFS and restored after a reload, keyed by model, engine build, context, slots, GPU/CPU, `swa_full` and template. LFM3 Q8_0 CPU, 678-token opening, 17 MB state: first call after a reload 5.5 s instead of 98 s, decision scores bit-identical to the saving load | no | no | Python native and SGLang: deliberately not built; the actors restart rarely and their in-process caches already reuse the opening |
| Page serves the stock model with the full sliding-window cache (`swa_full`, as the managed server's `--swa-full`) | yes | yes (`local-model.ts`, 2026-11-14) | n/a | n/a | none. Without it the page continued no earlier prompt past a ~170-token checkpoint, and a decision with a prompt beyond the sliding window aborted the engine (`failed to restore context checkpoint`, a `GGML_ABORT` in `server-context.cpp` when a checkpoint does not fit the SWA cache; also in the vendored natlang-b12074 build): from natlang-b12076 a checkpoint that does not load falls back to re-processing the prompt (fork cab27bfbe) |
| Native writes: `native.text-write`, `native.init-body`, `native.capture` (the learned operators' in-context capture, OPERATORS_RESTORE_20261117.md), `view`, `operator/1` | no direct native APIs; in-context capture is implemented by the shared HTTP `grammarCapture` client using produce spans | no direct native APIs; the engine can produce spans, but `BrowserLocalModel.neuraleseRuntime()` does not yet bind `capture` | yes (`engine.py` `native_capture`, `_native_write_text_impl`) | yes (`_sglang_native_capture` and `_sglang_write_text` through the fork's capture request; `native.init-body` still runs on the resident Python model) | managed HTTP capture client is implemented in `model/native-operator.ts` and wired by `native-state-client.ts`; managed fork/Python exact-span equivalence was measured live (two sites, 400-token spans; see evidence above). That check and the wllama produce-span smoke establish the engine primitive, not the production BrowserLocalModel capture binding. Page capture is blocked on the natlang-b12081 / wllama `5b0dcd35` public ABI: it exposes no tokenize/detokenize or native role-marker-ID query required by exact grammar capture (`vendor/wllama/esm/wllama.d.ts:180-439`; `model/native-operator.ts:27-65`). `BrowserLocalModel.neuraleseRuntime()` consequently has no capture binding. The active operators restoration (`a14187520`, `d306de499`, merged by `f577c13a8`) still needs those public fork/page primitives and a release through `update_llama_runtime.mjs --fork`; direct vendor edits or assumed token IDs are not equivalent. Other direct native write/init-body/view/operator APIs remain absent from the fork/page; grammar capture alone does not implement those APIs |

| Hybrid SWA KV, FP8 KV, mixed adapters in one batch | partly | partly | n/a | yes | none needed for CPU hosts |

Finite decision generation (2026-11-17): when no scored readout is available, the shared TypeScript agent sends the
same encoded decision messages through ordinary ModelTurn generation with `tools: []`. Managed llama-server, wllama,
Python native and SGLang use their existing generation route; no new serving field, probability approximation or
engine-specific implementation is introduced. Generated receipts bind generation graph nodes, not scored decisions.

Second reasoning close (2026-11-19): a reply that writes `</think>` again after its reasoning closed is answered by the
shared chat adapter (`ts-host/src/model/chat-completion.ts`, `secondThinkClose` in `native/tool-markup.ts`) with a
one-sentence error, nothing run. No engine changes: each splits the reply at its first close and returns the rest,
second close included, as content. Managed llama-server and the page: the fork's autoparser (`reasoning(until("</think>"))`
then content, `common/chat-auto-parser-generator.cpp` 137-167); Python native and SGLang: `serve/chat.py` `build_message`
(`partition("</think>")`) and the streaming split in `serve/http.py` (`tests/neuralese/test_lfm3_format.py`
`test_a_second_think_close_reaches_the_client_in_the_content`). The page loads its model with thinking off
(`browser/local-model.ts` `reasoning: false`: the prompt already closes an empty think block), so any `</think>` it
generates is a second one. The parser starts from the generation prompt's suffix from `<think>` (`p.prefix(generation_prompt,
reasoning_start)`, line 133), so the prompt's own `<think></think>` is the parsed (empty) reasoning block and a generated
close stays in the content there too (read from the code; not run against a page build).
