# Flint generation teacher replacement

The owner requested an actual test of RecursiveSI/flint and replacement of Qwen
if decent. The supervisor selected Flint for continuous generation after the first
18 matching completed native tasks: Flint passed 17, Qwen passed 15. Flint recovered
three cases Qwen did not finish successfully, while one Qwen success exhausted
Flint's execution budget. Most matching successful cases had shorter elapsed
execution with Flint. These are practical observations under each engine's serving
load and selected sampling settings, not an isolated throughput benchmark. The
remaining 48-case comparison continues and can change the operational decision.

Both runs use the same source-bound task keys and seeds, the existing natlang
executor and full trajectory capture. No held tasks or external teacher generation
were added. Successful collection is separate from verified conversion, training
admission, learner adoption and actual exposure.

## Actual deployment

- Existing teacher sandbox: `78530f71-08fc-4f57-ace9-7fbad64aeeb8`, boot
  `6b9f17ce-9809-432f-a7cc-51ffac393cf2`, RTX PRO 6000 Blackwell, 96 GB.
- Base: [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint), revision
  `2a247be0c0aec895ccb0ac7ca92229ec6df37288`.
- Quant: [official NVFP4 GGUF](https://huggingface.co/RecursiveSI/flint-nvfp4-gguf),
  revision `6ec43e19b565fc8d560e2d04985d7724f20673bf`; actual artifact
  22,863,084,160 bytes, SHA-256
  `97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
- Engine: [upstream Flint support](https://github.com/ggml-org/llama.cpp/pull/30928)
  at `1b44e30775dcb2f93952116b05f89e5bd797fb00`, merged with the upstream Jinja
  fix at `9febe63d5a379f8568bae7116a5b2ba844a4a4a6`; actual build
  `fe3e6ba71c885af4b58b4f1a53610d0f03acc716`, CUDA 13.4, sm120, Release.
- Eight serving slots, 65,536 tokens per slot, F16 KV, flash attention; temperature
  1, top-p .95, top-k 20, min-p 0. Actual reasoning content remains captured.
- Root: `/work/generation/flint-evaluation-20261021`; continuous generation under
  `production`, initial 64 TRAIN tasks, 16 collectors, then the existing broad
  qualified source sequence. No additional sandbox allocated.
- Serving intent SHA:
  `75b2a602263cd16281ed9eb49730883dc18dfbb65debc36ce7c7e467cba3c8ff`.
- Execution identity SHA:
  `b1a5ab2c050f3081c34ad710e7885b6190fe696023dbaaebd58aff62ac67d597`.
- Teacher source descriptor SHA:
  `5d6a0ee03aee5415ba8124f54c468d433f15ac6f6c57f055d78560018a22ce8f`.
- Guardian v2 preserves the collecting feeder and fixes its exact nice-to-Python
  exec observation. Its source is frozen separately from the failed v1 supervisor.
  Actual generation GPU utilization was 99% at 17:59 UTC.

## Continuation obligations

Finish scoring the remaining comparison cases and inspect the budget failure.
Maintain generation, verify private exports, convert and admit useful complete
trajectories through the shared mechanism for both students. Use the explicit Flint
namespace while retaining verified Qwen predecessors for deduplication. Do not
relabel existing Qwen batches or claim Flint exposure before actual adoption and
per-row training. Qwen batches 88–99 retain useful results and exact unfinished
accounting; reconcile their entire export inventory, rather than stopping at 88.

Increase serving concurrency or change other serving settings when measured useful
throughput supports it, preserving live cases at a natural boundary. New model
identity and sampling are immutable source configuration, not a separate data or
training pipeline. Shared decomposition:
`shared-state/TEACHER_MODEL_HANDOVER_DECOMPOSITION_20261021.md`.

Host receipts and operational controls are under `/home/werg/data/dgxops`:
`flint-continuous-native-teacher-activation-20261021.json` and
`flint-continuous-native-publisher-20261021/`. The publisher uses the explicit
CoreWeave SDK Python, not the project-only environment. Its durable spool is
`/mnt/external/natlang-development-data/teacher-incremental-flint-7853-20261021`.
