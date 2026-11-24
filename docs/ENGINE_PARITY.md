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

SGLang hosts each keep their own fork checkout (an editable install: `training/neuralese/sglang-fork.json` names the
commit, the actor attests the one it imports as `serving_engine.fork`). Pop's stayed at dbc51ddb72 (no tool grammar, no
FP8 weights: its actors sampled tool calls unconstrained) until 2026-11-24, when it moved to 17e3458c2e for its standing
collection actor; that actor's `/v1/neuralese/info` then matched the DGX collection actor's capabilities, routes and
prompt rendering identity, differing only in `max_running`. Check `serving_engine.fork.commit` on every host when the
pin moves.

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

## Numerics: SGLang scores depend on the batch and the KV dtype

Measured 2026-11-23 (DECISIONS "decision parity: FP8 KV, BF16 head logits, options' shared tokens"; sft-v8, A100,
vendor projection, 44 decision items). One request alone is bit-reproducible; the same ids batched with other
requests move an option's score by up to 1.66 nats with FP8 KV and 0.51 with BF16 KV, a radix hit that splits the
prefill by up to 0.2, and the actor's score schedule (`sglang_score`: one request per option, items concurrent) by
0.74-0.80 (FP8) or 0.26 (BF16). Against the native engine FP8 KV gives TV 0.038 mean / 0.27 max over a decision's
options, BF16 KV 0.012 / 0.051 (the native engine's own A100 vs GB10 gap is of that size). The e4m3 rounding of K/V
is 2.65% RMS per element (mantissa, not range: a calibrated scale does not reduce it). `--enable-deterministic-inference`
makes every schedule bitwise equal but turns the radix cache off with FlashInfer. The actors keep FP8 KV for decode
bandwidth; `--sglang-kv-cache-dtype auto` is the opt-in.

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
| 922903686 | `open_reply` in `/v1/natlang/score`: each continuation scored as the start of the reply (the final assistant message continued, `continue_final_message: "content"`, no end of message), echoed as `open_reply: true` in the result and the decision capture | every engine: section "Record decisions" |
| 251abbc38 | the release builds werg/wllama 05e08ad: the page's chat-template defaults (`default_template_kwargs`, LFM3 `preserve_thinking`) become the engine's own defaults at load, as the managed server's `--chat-template-kwargs`, so `/v1/natlang/score` renders items with them; before, only chat requests carried them and in-page readouts dropped the history reasoning | page only; the managed server passes `--chat-template-kwargs` from the release (c17f8e7e6), the Python engines render with the checkpoint's template |
| 550cb0313 | `/v1/natlang/score` cuts every continuation from the closed rendering: after its content (`open_reply`) or after its end-of-turn token. Before, a closed continuation also scored the template's newline after `<|im_end|>`, which the Python engines (`decision_items`) do not; and the continued rendering (922903686's first form) wrote the empty think block as `\n\n</think>\n\n` where the closed one writes `</think>` | every engine: the Python engines and SGLang score `</think>` + content (+ `<|im_end|>` when closed); checked on the fork with native_recording captures (closed 13 tokens ending `<|im_end|>`, open the same 12 without it). Closed scores of a decision move by the log-probability of a newline after the end of message (about 0 nats) |
| e528eb5a3 | `split_tokenization` cuts the common prefix and suffix at word boundaries (a word byte is ASCII `[A-Za-z0-9_]` or >= 0x80): `"spam"` vs `"ham"` share `am"` inside the word, which stays with it | every engine: the same rule in the Python engines and the learner |
| b4a202717 | `split_tokenization` in `/v1/natlang/score`: each option's probability sums its canonical tokenization and its split one (the options' common leading and trailing characters tokenized apart: `"spam"` also as `"`,`spam`,`"` where the canonical BPE writes `"sp`,`am"`); a split form equal to the canonical tokens counts once; results echo `tokenizations`, the capture records `continuation_alternatives` | every engine: Python native and SGLang (`decision_items` split_tokenization), learner (logsumexp over an option's sequences); client sends it by default (`NATLANG_DECISION_TOKENIZATIONS=canonical` opts out). prompt_logprobs servers (vLLM) score the canonical tokenization only: the chat route has no way to ask for another split |
| 73cc6a7be | a dispatched release builds what natlang installs (each platform's CPU archive, Metal, the page's wllama) and the quick GPU archives; Windows CUDA and ROCm (1.5-2 h each) run on a tag push or with the `full` input | build only: `update_llama_runtime.mjs --fork` pins CPU/Metal archives and the wllama archive |
| 7f89f2043 | the release workflow pins the prebuilt web UI assets (`HF_UI_VERSION`): a fork build number without its own bucket fell back to upstream's latest UI, which the tree's `ui-assets.cmake` rejects (natlang-b12082 failed) | build only; no engine serves the web UI (`--no-webui`) |
| c9fc1aba8 | the release's Windows CPU build defines `_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH` (the runner's MSVC STL 14.52 refuses its bundled Clang older than 22) | build only |
| af0f93100 | `/v1/natlang/score`: the tokens every reply of an item shares are prompt (they join `n_score_from` and the decision capture's prompt), each reply scores only its own, at least its last token; `NATLANG_DECISION_SHARED_PREFIX=option` keeps them | every engine: the Python engines and SGLang render through `decision_items` (`_shared_into_prompt`, 2026-11-23); checked on the natlang Q8_0 GGUF (CPU): closed and open items score the same own tokens per option as `decision_items`, captures equal |
| 0bee37915 | `/v1/natlang/score` items take `reasoning`: each continuation is the content of an assistant message with that reasoning (`reasoning_content`), so the reasoning is prompt; results echo `reasoning: true`, the decision capture records it | Python native and SGLang the same day (`native_learning.decision_items`); the page from the same release; gap row above until the pin moves |
| b704a8b2f | native-state files: `--native-state` and the page's `native_states` load natlang's store files (a release's `states/<id>.nzs`) through the PUT path (verified once, stored as read); PUT `{"path"}`; non-finite values refused; GGUF state-span format removed | every engine: section "Native state: one format, handed over as files" |
| 1a7f5015f | a text write or produce span is read in the reader's context (`in-context-text`): `producer_kind` ("text"/"span") is validated native-state metadata (refused with another value or with a `derivation`); a state of the server's identity carrying it, with no `adapter_binding` or `derivation`, is consumed as its own token ids inline with no consume markers, receipt `in-context-text`, recorded by the RL capture as tokens; every produced span carries `producer_kind: "span"` (its content id is the Python engines' for the same span); opt-outs `NATLANG_NEURALESE_TEXT_IN_CONTEXT=0`, `NATLANG_NEURALESE_SPAN_IN_CONTEXT=0`; `/v1/neuralese/info` lists the mode; an in-page reply carries its import receipts (`native_state.imports`) as an HTTP reply does | every engine: Python native and SGLang (`engine.py` `reads_in_context`, `_native_consume_text_in_context`; `sglang_actor.py` `_sglang_import`); the page compiles the same fork sources (werg/wllama builds `server-native-state.cpp`, `server-common.cpp`, `server-context.cpp`) |
| 89026ef24 | `/v1/natlang/score` `reasoning_ids`: a reasoned item's sampled reasoning ids replace the reasoning text's re-encoding, kept up to the first token the template writes after the reasoning (LFM3 `</think>`); ids need `reasoning` and must lie in the vocabulary (per-item error otherwise); results echo `reasoning_ids: true`, the capture records `reasoning_ids_kept`; split forms use the same ids | every engine: the Python engines splice the same ids (`serve/native_learning.decision_items`, SGLang through `_score_item`); the client sends them from the reasoning generation's RL capture (`native/agent.ts`), opt-out `NATLANG_DECISION_REASONING_IDS=rendered` |

## Record decisions: joint and field by field

A `readout: decision` over a finite record or a union of records (native/decision.ts `decisionDomain`) is scored by
the one client (`model/decision-scoring.ts`, `native/agent.ts` `decisionReadout`) on every engine. Up to 32 values,
every record is one closed continuation of one score item, as a flat decision is. Above, each step is one score item
whose continuations are open (`open_reply: true`): the reply's start through the next field, cut where the next key's
name starts. An engine must echo `open_reply: true` in each result of such an item; the client refuses one that does
not (`decision-open-unsupported`: the call generates its record instead), so an engine that ignores the field is never
read as having scored open replies.

| Engine | Open replies | How |
|---|---|---|
| Managed llama-server | from natlang-b12088 (fork 922903686, 550cb0313) | the closed rendering cut after the content |
| In-page wllama | from natlang-b12088 (the release's page build) | the same server code |
| Python native engine | yes | `serve/native_learning.decision_items(..., open_reply=True)`: the closed continuation without its end of message |
| SGLang actor | yes | the same items (the front renders, SGLang scores the ids) |
| prompt_logprobs servers (vLLM) | `continue_final_message: true` | not verifiable from the response; vLLM honours it |

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

**A text write or produce span is read in the reader's context (`in-context-text`, 2026-11-23).** `native.text-write`
encodes a value's text from position 0 with nothing before it (no BOS, no call), and a produce span holds the KV of its
producer's call; a reader that splices either cannot use it: 1 of 87 held read-backs returned the producer's text, and on
the probe a spliced span read back 0 of 8 on the native engine and on SGLang where the same text inline read 6 of 8
(DECISIONS 2026-11-23 "Neuralese read-back" and "marker rows, produce spans, producer kind"). The native-state header
therefore records how a state was produced: `producer_kind` 'text' (a text write) or 'span' (a produce span from
`native.capture`, the grammar capture or generation), optional and part of the content id, absent for every other
state. Every engine applies one rule: a state exact for the engine, with no `adapter_binding` and no `derivation`, whose
header has a producer kind and for which the request declares no state mode, is consumed as its exact token ids inline
in the reader's own context, with no consume markers. The receipt is `{"mode": "in-context-text",
"observed_native_state_id": id}` and the RL capture records the tokens, so learners replay what was computed. Python
native and SGLang: `engine.py` `reads_in_context`, `_native_consume_text_in_context`; `sglang_actor.py` `_sglang_import`
(`test_sglang_actor.py` `test_text_written_state_is_read_as_its_tokens_in_the_readers_context`,
`test_produce_span_is_read_as_its_tokens_in_the_readers_context`); a text write stored before the field existed is still
known from its store's `text-writes/<id>` mark. The managed fork and the page from natlang-b12092 (fork 1a7f5015f:
`native_state_registry::resolve`, every produced span carries `producer_kind: "span"`, the page compiles the same
sources). Opt-outs on the serving engine: `NATLANG_NEURALESE_TEXT_IN_CONTEXT=0` (text writes) and
`NATLANG_NEURALESE_SPAN_IN_CONTEXT=0` (spans) splice as stored. Contextual bodies, optimized, derived and soft states
still splice exactly on every engine. ts-host accepts the receipt for write-source imports, scoped states, implicit
read occurrences and native-gradient bindings (an in-context read binds no native input).
`scripts/neuralese_data/neuralese_readback_probe.py` is the acceptance probe (a text-written state, a captured span,
the same text inline, a shuffled state) against any front.

**Replies never sample a consume marker (2026-11-23).** The engine places `<|neuralese_consume|>` and
`<|/neuralese_consume|>` around an imported state; a sampled one is the protocol error
`neuralese-native-state-consume-marker`. With the marker rows initialised from text (their tied head rows no longer
the base's generic reserved row), a reader copying a spliced state's markers sampled one at greedy in 3 of 8 probe
reads. Every draw bans both (`engine.role_bans`): the Python native engine in its law (`_native_bans`) and SGLang by
logit bias (`sglang_banned`), both recording `banned_token_ids` per sample so learners score the same law
(`lower_rollout_turns` bans, `exp0_sft.law_mask`); the GGUF export's `serving.logit_bias` carries them for the managed
llama-server and the page (`native_gguf`). Opt-out `NATLANG_BAN_CONSUME_MARKERS=0`.

**The learned operators' in-context capture** (OPERATORS_RESTORE_20261117.md) has a managed HTTP implementation and an engine primitive shared by the fork and page; the page runtime binding remains blocked as recorded below. The Python engine and the SGLang actor serve `native.capture`: the actor goes through the fork's capture request, the request its forced-producing parity covers. The managed llama-server and the page serve no write endpoints and write states only from produce spans; the managed HTTP `grammarCapture` client sends the call's own messages again with a grammar of the exact ids `<reply prefix> produce_open <content> produce_close`.

**A capture whose prompt imported state** (an operator call under its own soft prompt state, 2026-11-20): the call's RL
capture records the import as a `native_state` prompt segment, and the capture keeps it in place on every engine. The
Python engine and the SGLang actor take `prompt_segments` in `native.capture` (the SGLang front sends the fork one
capture request with the state's rows as `natlang_import`; `test_sglang_actor.py` checks it against the Python engine's
capture tensor by tensor). The fork and the page re-send the call's own messages (the state as a `native_state` part) with
the forced span (`grammarCapture`). Each returns the source capture with the state as a `native_state` segment of
`token_count` rows. The learners read it as fixed stored context inside the producer document
(`lower_rollout_turns.write_sources`, `natlang_neuralese.train.lfm3_fast.passthrough_plan`). No gap.

**Merged BF16 export, in-place rebind and measured route attestation (2026-11-22, lfm3opti.md §6 item 4):** SFT
(`exp0_sft.py`) and GRPO (`grpo.py`) train the shared `lora+rows` core and publish merged BF16 weights
(`bf16(fp32(W0) + s * B A)`), so every engine serves a standard checkpoint: the managed llama-server and the in-page
wllama load its GGUF, the Python native engine its native checkpoint, the SGLang actor its HF export. Only the SGLang
actor takes in-place policy updates (GRPO's `WeightSync`); the other three serve one checkpoint per process. On a
rebind (`POST /v1/neuralese/rebind-policy`) the front reloads its native route from the same export, with every
native parameter matched by name (an unmatched name fails the reload). It drains native calls at a call boundary
first and invalidates the adapter replacement cache and prefix cache. `NATLANG_REBIND_NATIVE_RELOAD=0` opts out, and
the native route then attests the start. `GET /v1/neuralese/info` `routes` reports, measured, what each route serves:
`generation`, `decision` and `capture` give SGLang's own `/model_info` directory and weight version and the front's
SHA-256 of that weights file; `native_fallback` gives the reloaded model's `native_weights_identity`, or while not
resident the pending source and the claimed identity its lazy load verifies; `lineage` gives the start checkpoint's
identity, which exact native-state splicing keys on on every engine. `POST /v1/neuralese/sentinel` scores one token
sequence on SGLang and, when resident, the native engine; GRPO requires both to agree with the learner within
`grpo.SENTINEL_TOLERANCE` (`attest_all_routes` and `WeightSync._sentinel`). A plain engine reports its one
checkpoint on every route. No gap.

Live check, 2026-11-17: natlang-b12081 CPU with the natlang-lfm3-5ff07638 Q8_0 GGUF against the Python engine (BF16, CPU) on 5ff07638, two view sites, 400-token spans.
- The fork's rendered prompt equals its RL capture's prompt ids.
- The two states hold the same token ids and the same 38 tensors.
- Relative L2 difference: median 0.025-0.026 and largest 0.045-0.048 (layer-12 V), the Q8 floor. For comparison, SGLang vs Python forced producing differs by 0.050 (fork-checks).

## Tuned adapters (LoRA) on LFM3

A program fit (`tune.build`: native free-LoRA or subspace factors) serves on every engine. The Python native engine
applies the stack exactly (`native_adapters.run_native`) and is the reference. The SGLang front sums the stack into
resident slot factors per fused fork module (`sglang_actor._sglang_adapter_file`). The managed llama-server and the
page load the GGUF LoRA export (`tune/export.py` `export_peft` -> `export_gguf`, the vendored fork converter) at scale 0
and send per-request scales (`model/lora-scales.ts`); the fork's LFM3 graph applies it through `build_lora_mm` on
q/k/v, `attn_gate`, `attn_output`, conv `in_proj`/`out_proj` and `ffn_gate`/`ffn_up`/`ffn_down` (`src/models/lfm3.cpp`,
`lfm-base.cpp`). The export carries every target the tuner trains: on this release 55 attention, 26 conv and 72 MLP
tensors, the attention gate and `o_proj` included.

Measured 2026-11-24 (T3) on the first real LFM3 program fit, `ext_decisions_route` (free LoRA rank 16 over every LFM3
family, base 5ff07638). The prompts are 6 of the program's rows rendered with the serving template, 4.9-6.9k tokens
each, run greedily for 24 tokens with the adapter off and on. Each engine's tokens are scored by the native model on
that engine's own path (float32 for llama.cpp, BF16 for SGLang). Harnesses:
`scripts/neuralese_data/check_tuned_lora_llama_parity.py RELEASE`, `ts-host/scripts/wllama-lora-smoke.mjs --parity`,
`scripts/neuralese_data/check_sglang_adapter_parity.py --release`.

| Engine | How served | Greedy agreement (positions where the engine's token is the native argmax) | mean / max \|Δ logprob\| of the engine's tokens | Adapter applied |
|---|---|---|---|---|
| Python native engine (reference) | `run_native` on the release's `.nnt` stack, BF16 (the identity the release binds: a float32 engine refuses it as another live checkpoint) | reference | reference | fit targets' mean NLL 1.49 -> 0.32 nats with the adapter |
| Managed llama-server, natlang-b12088 linux-arm64 CPU (`src/` and `ggml/` identical to the pinned b12092) | f32 GGUF of 5ff07638 + the release's GGUF LoRA, `--lora-init-without-apply --lora-per-sequence`, per-request `lora` scale 0 / 1 | 288/288 (6 prompts x 24 tokens, adapter off and on): every continuation equals the native greedy one | 0.004 / 0.21 (the BF16 reference's rounding) | 6/6 continuations change; the adapter moves the native logprob of these tokens by 1.90 nats on average |
| In-page wllama (vendored natlang-b12092, headless Chromium, CPU) | Q8_0 GGUF + the same LoRA blob at scale 0, per-request `lora` | 144/144 (3 prompts x 24 tokens, off and on) | 0.006 / 0.11 | 3/3 change (1.69 nats); text prompts tokenize to the prompt's ids (llama.cpp `/tokenize`); tokens equal the managed server's f32 run |
| SGLang actor slots (fork 17e3458c2e) | the front sums the stack into resident slot factors (`custom_params.natlang_adapter`), BF16 | before the fixes below (GB10, 2 running): top-1 agreement 0.34 adapted / 0.78 base | 2.6 / 0.70 nats adapted / base | control (native base vs native adapted on the same tokens) 1.79 nats. Blocked: the keyed re-measurement crashed in the fork (CUDA illegal memory access in decode after a mixed adapted/base prefill at 4 running requests, on the GB10 and the A100); a run at 2 running is queued on the A100 (`runs/t3-score/sglang-keyed`, with a CUDA_LAUNCH_BLOCKING trace in `sglang-debug/collect.log`) |

The export is exact: the PEFT's dense deltas equal the native stack's low-rank deltas on all 153 targets (max relative
gap 0.0).

Two defects found on the SGLang path, both fixed:

- **Slot rank.** The front packs q|k|v into one fork module (`qkv_proj`) and w1|w3 into another, so a rank-r fit over
  every LFM3 family needs 3r in a slot. Slots of the default rank 32, and of the fit's own rank (passed by
  `pop_actor_driver` and the Colab measurement actor), were too small, and every adapted turn ran on the native engine
  (the first run had 12 of 24 requests on SGLang, none of them adapted). `serve_native_checkpoint.py` now sizes the slot
  from `--adapter-release` (`release_slot_rank`, `native_hf.packed_slot_rank`: 48 here). Pop's job and the Colab
  measurement actor pass the packed rank, computed on the DGX.
- **Prefix cache across adapters.** SGLang matched cached prefixes (KV and conv/recurrent state) on token ids and
  `extra_key` only, and the slot adapter (`custom_params.natlang_adapter`) was not part of the key. So a prompt state
  computed under one stack, or under none, served requests with another. With an adapted and a base request on each
  program prompt, base turns were 0.70 nats from the native engine (top-1 agreement 0.78) and adapted turns 2.6 nats
  (0.34). `sglang_actor.adapter_cache_key` now names the stack in `extra_key`, as SGLang's own LoRA does with
  `lora_id`; `NATLANG_SGLANG_ADAPTER_CACHE_KEY=0` opts out. The managed server and the page never reuse a prompt
  computed under other adapters (fork `--lora-per-sequence`, the vendored wllama patch). The Python engine's prefix
  cache keys on the execution binding (`engine.py` `prefix.digests({'adapters': execution_binding(self), ...})`).

## Result schemas: tuples

A function's result type reaches every engine as the `value` schema of its `return_result` tool (`native/agent.ts`
`schemaOf`). A tuple `[A, B]` is one JSON Schema 2020-12 `prefixItems` schema with `minItems` = `maxItems` = its
length. An optional tail `[A, B?]` is `anyOf` one such exact tuple per admitted length (`[A]`, `[A, B]`): llama.cpp's
grammar writes every `prefixItems` element and ignores `minItems`/`maxItems` there (`common/json-schema.cpp`
`build_array` returns a fixed `common_chat_schema_tuple`; `json-schema-to-grammar.cpp` `KIND_TUPLE`), so the single
schema `{prefixItems: [A, B], minItems: 1}` forced B. No schema emits `items: false` beside `prefixItems`: the fork reads
`items` before `prefixItems` and refuses a boolean schema. `NATLANG_TUPLE_SCHEMA=prefix-items` keeps the single schema.
Whatever an engine enforces, the runtime checks the returned value against the type (`native/values.ts` `coerce`:
arity, an omitted tail, each element), and a call the client parses from text (`model/native-xml-calls.ts`, program
headers) validates `prefixItems` by position and `items` after it.

| Engine | Tuple result schema | Evidence |
|---|---|---|
| Managed llama-server (fork cb03daee7 / 0bee37915, natlang-b12081 / b12088: same schema code) | enforced: the autoparser's lazy tool grammar (`chat-auto-parser-generator.cpp`, `tool_choice: auto`, trigger `<tool_call>`) constrains the `value` parameter to the schema | `scripts/wllama_smoke.mjs` tuple leg, 2026-11-23, natlang-b12081 linux-arm64 CPU, LFM3-1.7B Q8_0 with its release template, greedy, thinking off: asked for `["red", "green", "blue"]`, `[string, number]` and `[string, number?]` return `["red", 2]`; asked for `["red"]`, `[string, number?]` returns `["red"]`; the unconstrained control (`value: {}`) returns the three strings. With `NATLANG_TUPLE_SCHEMA=prefix-items` the `["red"]` request returns `["red", 1]` (the forced tail) |
| In-page wllama (natlang-b12081 build) | enforced, the same server code | the same smoke, the page leg: the same values, `["red", 1]` under the opt-out too |
| Python native engine | enforced (2026-11-23, `common/tool_grammar.py`; `NATLANG_TOOL_GRAMMAR=0` or a request's `x_natlang_tool_grammar: false` opts out): every reply to a request with tools decodes under one xgrammar structural tag built from the tools its prompt renders (free text unconstrained; a `<tool_call>` names a declared tool and its parameters follow that tool's schema in the backbone's call format, `qwen_xml` for LFM3). `Engine._native_token` masks the logits before guidance bans, temperature and the nucleus; the matcher follows every reply token (`_native_record_input`) and a guidance rewind (`native_guidance.check`); a templated `decode` opening is fed first, and one the grammar refuses leaves that reply unconstrained (`natlang_tool_grammar_refused_openings_total`). The capture declares the law in `sampling_law.tool_grammar` | `scripts/neuralese_data/check_tool_grammar.py decode`, 2026-11-23, sft-v9 (e15b20a8) BF16 on the DGX, greedy, reasoning on (the natlang template): `[string, number]` and `[string, number?]` asked for `["red", "green", "blue"]` return `["red", 2]`; `[string, number?]` asked for `["red"]` returns `["red"]`; the unconstrained control (`value: {}`) returns `["red", "green", "blue"]`; `tests/neuralese/test_tool_grammar.py` |
| SGLang actor | enforced from fork 17e3458c2e (`NATLANG_GRAMMAR_PREFIX`): the front sends the same structural tag (`sampling_params.structural_tag`), the fork builds its xgrammar backend under `--skip-tokenizer-init` from the served tokenizer and applies the mask before sampling and before its natlang records, so the recorded logp is the masked law; a continuation request (a guidance redraw, a forced feed) carries `natlang_grammar_prefix`, the reply tokens its grammar takes first. The front follows the reply with its own matcher and refuses a token outside the grammar; the actor law (`serving_engine.tool_grammar`) records it. A fork without the marker serves unconstrained | the same decode check on the SGLang front over fork 17e3458c2e (a ledgered actor, 2026-11-23): the same four values; both engines declare the same grammar sha256 per tool list; `serving_engine.tool_grammar` true; fork tests `test/natlang/test_grammar_prefix.py` |

**The training law of a constrained reply.** A grammar mask applied before sampling changes the law a token is drawn
from, and both training-collection engines record the masked, renormalized log-probability (the Python engine masks
before `masked_log_distribution`; SGLang masks in `model_runner._preprocess_logits`, before the fork's natlang
records). The capture declares the grammar (`sampling_law.tool_grammar`: the structural tag, its sha256, the xgrammar
version, the tokenizer identity the masks were computed over); a mask is a function of the grammar and the reply's
tokens before it, so the learner recomputes every one exactly (`tool_grammar.replay_masks`, the reply's forced opening
first) and scores each target under it, the policy, the reference and hence rho and the KL alike
(`lower_rollout_turns.sampled_call` keeps the law and guidance bans, `exp0_sft.law_mask`, `lfm3_fast.HeadMask` in the
fused head NLL and nucleus, used by `grpo.py` and `exp0_sft.py`). A law recorded over another tokenizer or compiler
version is refused, not scored unmasked. The whole-graph trainer (`rl/native_replay.py`) refuses a capture whose law
has keys it cannot replay (`native RL sampling processor needs explicit replay support`), this one included.
Measured 2026-11-23 (`check_tool_grammar.py learner`, the captures of the decode check, sft-v9 at its own weights):
Python engine captures: calls 9, 4,406 sampled tokens, 270 constrained (the grammar allows fewer than the 125,037 real tokens free text allows). With the mask, log rho = learner - recorded has mean -0.0003 on the constrained tokens (|max| 0.20, BF16 engine vs learner noise) and -0.0002 on free text; unmasked, its mean on the constrained tokens is -0.979 (min -26.2 nats: tokens the grammar forced that the unmasked policy found unlikely), the bias a learner without the mask would have put on every constrained token; free-text tokens are unchanged by the mask (the padding rows it removes carry no measurable mass). SGLang captures declare the same grammar; their learner leg is not measured yet (DECISIONS).

## Open gaps

| Feature | Fork | Page | Python native | SGLang | Next action |
|---|---|---|---|---|---|
| Decision: reuse the function's opening across inputs (`cache_messages`, fork e51ca6d68) | yes | yes from natlang-b12074 | equivalent: the prefix cache resumes from any stored grid block | yes, to within 255 tokens (2026-11-14, `sglang_actor._sglang_warm_opening`, on by default, `NATLANG_SGLANG_WARM_OPENINGS=0` opts out): with `extra_buffer` the fork (ace8d6868) snapshots conv state once per prefill, at `prefix + floor(extend / 256) * 256` (`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`; grid lcm(LFM3 `mamba_chunk_size` 256, page size)), and a finished request inserts its tokens only up to that snapshot (`MambaComponent.prepare_for_caching_req`); a request resumes only from a state on its own path, so each input's state lies inside its own input, the second input re-prefills the opening (only it tracks a branching state there, `finalize_match_result_in_tree_core`), and concurrent items of a batch all do. The actor therefore renders the first `cache_messages` messages, checks they are a token prefix of the prompt, and sends one prefill request of exactly those tokens per distinct opening (concurrent items wait for it; re-sent after 60 s or a weight update) before any option is scored: each input then prefills at most 255 opening tokens plus its own. Exact from the next pin, fork 33230160b (`training/neuralese/sglang-fork.json`): the warm request sets `custom_params.natlang_track_input_end`, the fork's scheduler cuts its prefill so the last extend is a whole number of grids and the state lands on the boundary (`PrefillAdder._natlang_input_end_extend`), and every input prefills only its own messages; the front detects the fork's `NATLANG_TRACK_INPUT_END` marker and keeps the grid-aligned warm against older pins. An opening shorter than 256 tokens stores no state on either | at the next pin on the DGX, read `natlang_sglang_cached_prompt_tokens_total` / `natlang_sglang_prompt_tokens_total` (and `natlang_sglang_opening_warms_total`) over decision calls of one function with different inputs (on fork 33230160b, cached tokens should reach the full opening) |
| natlang's GGUF as the public playground's model (native states on the public page) | n/a | the local page: yes (`model-default.ts`, natlang-lfm3-5ff07638); the public page: no, it keeps the stock LFM3 Q4_K_M from Hugging Face (`build-playground-site.mjs`) | n/a | n/a | blocked on a host: the GGUF is a GitHub release asset, and neither `github.com/.../releases/download` nor its redirect target sends `Access-Control-Allow-Origin`, so a page on another origin cannot fetch it (the site smoke's `--network` check failed, 2026-11-16). A CORS-capable host (e.g. a Hugging Face model repo, as the stock model uses) is the owner's call; then point `build-playground-site.mjs` at `BROWSER_MODEL_RELEASE` |
| Training guidance turn (`x_natlang_privileged_context`, `x_natlang_guidance_policy`; capability `training-guidance`) | no: refused by ts-host (`training-guidance-unavailable`, `model/neuralese-server.ts`), never dropped | no: refused (`browser/local-model.ts`) | yes: renders the runtime's guidance message (`natlang.guidance-request/2`, ts-host `guidanceMessage`) with its template after the leading system message and marks its span (`engine.py` `_guidance`, rl/privileged_context `message_span`) | yes (inherits `_guidance`, `sglang_actor.py:619-621`) | The guidance turn has one definition, the runtime's system message (2026-11-18). Blocked on the fork's capture: it marks privileged tokens only from a pre-tokenized `x_natlang_privileged_block {insert_at, token_ids}` (`tools/server/server-rl-capture.cpp:43-54`), cannot place a writer state in it, and returns no guidance receipt. Next: a fork release (`update_llama_runtime.mjs --fork`) that renders the request's guidance message after the leading system message, flags its span and echoes the receipt, as `message_span` does; the page also needs that release. |
| Decision: score items of one batch concurrently | yes | yes (one engine request) | serial (one execution lock) | yes (2026-11-14) | none |
| Decision: one prefill shared by all options | yes (parent slot checkpoint) | yes | yes | yes: the first option warms the radix cache, the rest run in parallel (each option once since 2026-11-14) | none |
| Decision: options scored after the reply's sampled reasoning (score item field `reasoning`, the reasoned `readout: decision`; DECISIONS 2026-11-23 "reasoned decisions") | fork 0bee37915 on `neuralese` (token ids equal to the Python reference on 3 items, with `open_reply` too); in the first release built from it (release run 45120615947 queued 2026-11-23 behind the open_reply release run 45117905605) | from the same release (the page scores through the fork's endpoint) | yes (`native_learning.decision_items` `reasoning`) | yes (same rendering; `sglang_score` takes the plan) | move the runtime pin to that release (`scripts/update_llama_runtime.mjs --fork`). Until then the managed server and the page answer without the `reasoning: true` echo: the client refuses the scores (`decision-reasoning-unsupported`), traces it and reads that driver's decisions out instantly (`readout: instant` semantics) |
| Decision: tokens every option shares are prompt, each option scores only its own (2026-11-23; `</think>`, the record's opening, a field step's chosen fields; `NATLANG_DECISION_SHARED_PREFIX=option` opts out) | no: blocked on a fork release, fork af0f93100 on werg/llama.cpp-neuralese branch `natlang-shared-prefix`, not yet on `neuralese` (`server-context.cpp`: the replies' common token prefix joins `n_score_from`, at least one own token each) | no: the same release | yes (`native_learning._shared_into_prompt`; sft_sources rows carry them as `prefix`, `exp0_sft.decision_parts`) | yes (renders through `decision_items`) | merge af0f93100 into `neuralese`, release it and move the pin (`update_llama_runtime.mjs --fork`); until then the fork's option scores carry the shared tokens' log-probability (about -16 nats for LFM3's `</think>`), the same in every option of an item, which every readout cancels |
| Decision: the history renders with the release's template switches (LFM3 `preserve_thinking`) | yes from 2026-11-23 (`llama/server.ts` passes the release's `chat_template_kwargs` as `--chat-template-kwargs`: the score endpoint renders items with the server's defaults) | no: blocked, the vendored wllama applies `default_template_kwargs` to chat completions only (`vendor/wllama/esm/index.js` `createChatCompletion`), so `createNatlangScore` renders a readout's opening assistant turn as `<think></think>` | yes (`serve/chat.py` `bind_history_reasoning`) | yes | a fork release whose score endpoint takes the request's `chat_template_kwargs`, or a wllama build that sends `default_template_kwargs_keys/vals` at load |
| Head logits in FP32 (2026-11-23) | yes (llama.cpp's F32 logits) | yes | yes (`shared_state.logits`; the learner's head in `lfm3_fast`; `NATLANG_HEAD_LOGITS=bf16` opts out) | yes (`--enable-fp32-lm-head`, `serve_native_checkpoint --sglang-head-logits`, attested `serving_engine.head_logits`) | none |
| Score turns with generating slots (fork 0da69f720, ff6e734e3) | yes | n/a (one request at a time) | n/a | SGLang's scheduler; unmeasured | measure decision latency under collection load |
| Prefix-cache hit telemetry | upstream counters | n/a | `natlang_prefix_cache_hits/misses` | `natlang_sglang_prompt_tokens_total` and `natlang_sglang_cached_prompt_tokens_total` (2026-11-14) | none |
| Warm openings kept across restarts | yes (`--slot-save-path`) | yes from natlang-b12076 (`browser/warm-openings.ts`, on by default for a model loaded with its `sha256`; werg/wllama 0095d0f, `capabilities.slotState`): the opening is prefilled once, its slot state saved to OPFS and restored after a reload, keyed by model, engine build, context, slots, GPU/CPU, `swa_full` and template. LFM3 Q8_0 CPU, 678-token opening, 17 MB state: first call after a reload 5.5 s instead of 98 s, decision scores bit-identical to the saving load | no | no | Python native and SGLang: deliberately not built; the actors restart rarely and their in-process caches already reuse the opening |
| Page serves the stock model with the full sliding-window cache (`swa_full`, as the managed server's `--swa-full`) | yes | yes (`local-model.ts`, 2026-11-14) | n/a | n/a | none. Without it the page continued no earlier prompt past a ~170-token checkpoint, and a decision with a prompt beyond the sliding window aborted the engine (`failed to restore context checkpoint`, a `GGML_ABORT` in `server-context.cpp` when a checkpoint does not fit the SWA cache; also in the vendored natlang-b12074 build): from natlang-b12076 a checkpoint that does not load falls back to re-processing the prompt (fork cab27bfbe) |
| Consume-marker ban on the managed llama-server and the page | from the next GGUF export (`native_gguf` `serving.logit_bias`) | same, through the published model's `request.logit_bias` | yes | yes | Blocked on a GGUF export: the published natlang-lfm3-5ff07638 GGUF and its manifest predate the ban (`model-default.ts` `logit_bias: {}`); the next `native_gguf` export and `publish_browser_model.mjs` carry it. |
| Native writes: `native.text-write`, `native.init-body`, `native.capture` (the learned operators' in-context capture, OPERATORS_RESTORE_20261117.md), `view`, `operator/1` | no direct native APIs; in-context capture is implemented by the shared HTTP `grammarCapture` client using produce spans | no direct native APIs; the engine can produce spans, but `BrowserLocalModel.neuraleseRuntime()` does not yet bind `capture` | yes (`engine.py` `native_capture`, `_native_write_text_impl`) | yes (`_sglang_native_capture` and `_sglang_write_text` through the fork's capture request; `native.init-body` still runs on the resident Python model) | managed HTTP capture client is implemented in `model/native-operator.ts` and wired by `native-state-client.ts`; managed fork/Python exact-span equivalence was measured live (two sites, 400-token spans; see evidence above). That check and the wllama produce-span smoke establish the engine primitive, not the production BrowserLocalModel capture binding. Page capture is blocked on the natlang-b12081 / wllama `5b0dcd35` public ABI: it exposes no tokenize/detokenize or native role-marker-ID query required by exact grammar capture (`vendor/wllama/esm/wllama.d.ts:180-439`; `model/native-operator.ts:27-65`). `BrowserLocalModel.neuraleseRuntime()` consequently has no capture binding. The active operators restoration (`a14187520`, `d306de499`, merged by `f577c13a8`) still needs those public fork/page primitives and a release through `update_llama_runtime.mjs --fork`; direct vendor edits or assumed token IDs are not equivalent. Other direct native write/init-body/view/operator APIs remain absent from the fork/page; grammar capture alone does not implement those APIs |

| Gradients (`natlang:learning`: `POST /v1/neuralese/grad` recorded objectives and derivative programs, native-state and adapter optimizer steps) | no: blocked, llama.cpp has no backward pass through the model or the spliced native state | no: blocked, wllama has no backward pass; the page refuses the module (`ModuleUnavailableError`, `browser/index.ts`) and takes trained states and adapters as `.nz` files | yes (`serve/http.py` `/v1/neuralese/grad`, `serve/native_grad.py`, `serve/native_derivative.py`); conformance: `conformance/neuralese/cases/termination-learning-graph.yaml` `grad-*` and `graph-replay-fixed-choices`, run by `ts-host/scripts/native-conformance.mjs` against `tests/neuralese/learning_conformance_server.py` (2026-11-23) | no: blocked, the SGLang actor serves generation only (no gradient replay endpoint; its weights update by `rebind` from the learner) | train on the Python native engine and hand the results to every engine as native states or adapter files; an engine gains the row with a backward pass |
| Tool arguments decoded under their schema, and a captured reply recording the grammar it was drawn under (result types, tuples included; section "Result schemas: tuples") | decoded: yes (autoparser tool grammar); captured: no, `x_natlang_rl_capture` drops the tool grammar (`server-rl-capture.cpp` `rl_sampling_law`), so a captured reply is unconstrained and its recorded law is the unmasked one the learner scores | the same as the fork | yes, captured as `sampling_law.tool_grammar` | yes, the same | blocked on the grammar engine: the learner replays xgrammar's masks exactly (`tool_grammar.replay_masks`), and llama.cpp's grammar is another implementation of the schema (its token masks are not xgrammar's, e.g. whitespace and string rules), so a fork capture constrained by it could not be replayed exactly. Next: the fork compiles the same structural tag with xgrammar's C++ core in the server and the page build (a fork change and release through `update_llama_runtime.mjs --fork`); no training source reads fork or page captures today (RL, SFT and held collection run on the Python engine and SGLang) |
| Hybrid SWA KV, FP8 KV, mixed adapters in one batch | partly | partly | n/a | yes | none needed for CPU hosts |
| Tuned adapter (program-fit LoRA) parity on LFM3 | yes (2026-11-24: greedy 288/288 vs native) | yes (144/144) | reference | blocked: fork crash (CUDA illegal memory access in decode after a mixed adapted/base prefill at 4 running requests; GB10 and A100) and the re-measurement after the slot-rank and prefix-cache fixes (9cf56138e) not yet run | read `runs/t3-score/sglang-keyed` (queued A100 tenant) and its CUDA_LAUNCH_BLOCKING trace; localize the fork's slot path (`models/lfm3.py` `NatlangAdapters.token_mask` / decode graph variant) and fix it in werg/sglang-natlang |

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

### Source graph loading and generated module ownership

The project compiler, named-function loader and callable-folder loader use one synchronous source-load session
(`runtime/source-session.ts`). The session shares completed named records and parsed portable type aliases within
that immutable input, then releases its cache. A later load re-reads its source revision; private callable children
retain their lexical owner. `SourceFiles.sourceGraphReuse=false` or `NATLANG_SOURCE_GRAPH_REUSE=0` opts out.

Reachable public NL dependencies receive independent declarations rather than repeated inline copies of their
entire hierarchy. The existing `inlineUses` compiler option / `NATLANG_INLINE_USES_DECLARATIONS=1` retains the
reference declaration mode. Emitted NL wrappers and inline callable contexts import an owner-local generated record
graph; shared aliases are transported once and reconstructed into separate record maps. Only serializable frontend
products can be shared by Host's `NativeCompilerProducts`; runtime services, module instances and model state are
never shared. `inlineRecords:true` / `NATLANG_INLINE_SOURCE_RECORDS=1` retains inline record emission.

These mechanisms are common compiler/runtime code used by Node, Workerd, browser/wllama, Pi and llama CLI SDKs.
Managed llama-server, Python native and SGLang receive model requests from that runtime and do not load source
records or implement compiler emission. The five Vibe SDK vendors must stage the same Core revision. Actual compiled
Node and Workerd tests exercise diamond dependencies, inline contexts, source edits, lexical types, cycle rejection,
record opt-outs and exact native child lineage; an actual Node ESM import covers the separate ESM graph module.
