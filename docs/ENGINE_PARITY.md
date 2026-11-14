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
  same ranking on 3 items, largest gap per item 0.58-0.67 nats from native, a repeat within 0.16 nats.

## Numerics: a score is reproducible only under a fixed cache layout

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

## Open gaps

| Feature | Fork | Page | Python native | SGLang | Next action |
|---|---|---|---|---|---|
| Decision: reuse the function's opening across inputs (`cache_messages`, fork e51ca6d68) | yes | yes from natlang-b12074 | equivalent: the prefix cache resumes from any stored grid block | unknown: the radix cache keeps conv state only at its `extra_buffer` track points | at the next pin on the DGX, read `natlang_sglang_cached_prompt_tokens_total` / `natlang_sglang_prompt_tokens_total` over decision calls of one function with different inputs; if the opening is not reused, warm a request that ends at the hinted boundary |
| Decision: score items of one batch concurrently | yes | yes (one engine request) | serial (one execution lock) | yes (2026-11-14) | none |
| Decision: one prefill shared by all options | yes (parent slot checkpoint) | yes | yes | yes: the first option warms the radix cache, the rest run in parallel (each option once since 2026-11-14) | none |
| Score turns with generating slots (fork 0da69f720, ff6e734e3) | yes | n/a (one request at a time) | n/a | SGLang's scheduler; unmeasured | measure decision latency under collection load |
| Prefix-cache hit telemetry | upstream counters | n/a | `natlang_prefix_cache_hits/misses` | `natlang_sglang_prompt_tokens_total` and `natlang_sglang_cached_prompt_tokens_total` (2026-11-14) | none |
| RL capture under the sampling law (temperature 1, top-p 0.95) | yes from natlang-b12074 (dc9410950: host-side nucleus draw, untruncated logp) | n/a | yes | yes (untruncated logp recorded) | run the fork's `test_natlang_rl_capture.py` with `LLAMA_NATLANG_GGUF` set to a natlang export before the fork collects RL data |
| Warm openings kept across restarts | yes (`--slot-save-path`) | no (page reload starts cold) | no | no | low priority: actors restart rarely |
| Hybrid SWA KV, FP8 KV, mixed adapters in one batch | partly | partly | n/a | yes | none needed for CPU hosts |
