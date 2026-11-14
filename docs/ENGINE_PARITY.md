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
  (ranking, 1-nat tolerance, a repeat after a batch bit-identical); run it on every pin bump. natlang-b12074: same ranking on
  3 items, largest gap per item 0.58-0.67 nats from native, repeats identical.

## Fork commits reviewed for parity

| Commit | Change | Other engines |
|---|---|---|
| a0c92ce36, 0da69f720, ff6e734e3, e51ca6d68 | decision scoring from a shared-prefix checkpoint, score batching with generating slots, `cache_messages` | rows above |
| a6401bbd3 | streamed reasoning of a guided reply | row above |
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
| Streaming the open reasoning of a guided reply (fork a6401bbd3) | yes | yes from natlang-b12074 | no (sent at the end) | no | low priority for RL; add when an interactive host serves from Python |
| RL capture under the sampling law (temperature 1, top-p 0.95) | yes from natlang-b12074 (dc9410950: host-side nucleus draw, untruncated logp) | n/a | yes | yes (untruncated logp recorded) | run the fork's `test_natlang_rl_capture.py` with `LLAMA_NATLANG_GGUF` set to a natlang export before the fork collects RL data |
| Idle-slot prompt cache (`cache_idle_slots`) exact on wasm32 | yes | off (`local-model.ts`): with it on and 4 slots, a decision scored after a batch differs by up to 0.5 nats from the same decision before it (native bit-identical under the same flags; exact with it off or with 1 slot; not the thread count, not `swa_full`) | n/a | n/a | find the wasm32 divergence in `server_prompt_cache` save/restore of the hybrid (conv + SWA) state with `scripts/wllama_smoke.mjs` and `cache_idle_slots: true`, then turn it back on |
| Warm openings kept across restarts | yes (`--slot-save-path`) | no (page reload starts cold) | no | no | low priority: actors restart rarely |
| Hybrid SWA KV, FP8 KV, mixed adapters in one batch | partly | partly | n/a | yes | none needed for CPU hosts |
