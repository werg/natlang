# Automated behavioural evaluation on Pop (operator note, 2026-11-03)

Fallow register items #6, #12 and #20. Every newly earned LFM3 checkpoint gets a protected dev16 evaluation on the
otherwise idle Pop GPU and a paired comparison against the current behavioural parent, with no manual steps. Training
never waits on it, and it spends no DGX GPU time (owner "frugal eval": Pop is idle, so this compute is free).

## Service

- DGX ledger unit **`pop-behavior-eval`** (class service, priority low, 4 GiB budget, resumable) runs
  `scripts/neuralese_data/pop_behavior_eval.py watch --poll 300`. Restart it through the ledger with a fresh unit
  name if it stops; it resumes its in-flight candidate from `state.json`.
- State and evidence live in `W/ops/pop-behavior-eval/` (`W=/home/werg/data/lfm3-rl-dgx-20261029`):
  `state.json` (queue, parent, per-runtime parent receipts), `declaration.json`, `behavioural-parent.json`,
  `evidence/<arm>-<step>-<sha12>-py<runtime12>/` (completed.json, launch.json, detail.json), `comparisons/`,
  `selections/`, `preparations/`, `transfers/`, and retirement records for pruned views.
- `pop_behavior_eval.py status` prints the parent and every candidate's state and scores.

## Triggers

1. **Window boundary.** A controller `W/<lineage>/state.json` whose `last_learner_progress.checkpoint_sha256` changes
   to a SHA not yet seen. The runtime is the lineage config's `actor_environment.NATLANG_REPO`
   (`W/ops/<lineage>/config.json`). Controllers present when the watcher first started are history.
2. **Marked for eval.** `pop_behavior_eval.py request CHECKPOINT --runtime RUNTIME [--sha256 S] [--step N] [--label L]
   [--no-ema] [--attempt LABEL]` writes `requests/<ms>.json`. Requested candidates are never dropped as superseded.
   A request is idempotent only for the same checkpoint SHA under the same runtime and attempt label: it is then
   logged as `request-duplicate` with its disposition and not queued. Another runtime, or an explicit `--attempt`
   label, queues a fresh candidate (state key `SHA#label`, results labelled `-t<label>`), whose launch must be a new
   container.

## Execution identity (2026-11-09, review F8/F9)

Result labels, Pop work directories, the parent-evaluation cache (`state.json` `parent_evals` key
`parent:python:identity16`) and promotion all use one identity: `execution_identity()` hashes the Python runtime,
the frozen runtime closure (`runtime-input-closure.json`), `task_eval_runtime_identity` of the runtime (full TS host
closure and every executing Python module, resolved from the frozen runtime, never the checkout), the driver script,
the task spec/packet and the Pop image (name and resolved id). `collect` checks Pop's `launch.json` against the
requested identity and records `execution.verified`; `train/promotion.compare` takes both arms' `execution` and
reports unmatched (never promotes) unless both are verified and equal. Results recorded before identities existed
are not reused or paired.

One evaluation runs at a time. When a newer window-boundary candidate is queued, older queued ones are recorded as
superseded. A running candidate's EMA arm is skipped when a newer candidate is waiting.

## What one candidate runs

1. **View.** `pop_eval_prepare.py raw` runs as a low-priority ledger unit (8 GiB) inside the checkpoint's own frozen
   runtime (`--workdir`, `NATLANG_REPO` and `PYTHONPATH` all set to the runtime). It calls
   `prepare_actor_transport(scope='diagnostic-local-pin')` and reads the source in place: no copy, no pin, no
   hardlink. It prefers the learner's existing `.checkpoint.pt.actor-view-source-<sha>` pin when one exists. The
   source stat is checked before and after the read.
2. **Pop room.** Before sending, it prunes old model-only views and delta caches under Pop's
   `matched75-transport-0f8a-1101/` until the view plus an 8.5 GiB scratch floor fit. It always keeps the behavioural
   parent's stage, the peer delta base and any stage with a consumer lease. Pop has about 17 GiB free.
3. **Transfer.** `scripts/coreweave/send_actor_transport.py` (in process) sends a ZIP delta against the last view Pop
   received, or the full view when no base exists. The receiver reconstructs the view and rereads it before
   publishing.
4. **Runtime on Pop.** A runtime missing on Pop is installed at the same path from a tar stream. It is published only
   after its complete file inventory matches the DGX bytes.
5. **Evaluation.** `pop_dev16_driver.py` runs under Pop's memory ledger (unit `popeval-<arm>-<sha12>-<t>`, container
   of the same name). It uses the baseline contract unchanged:
   - task spec sha `f28b1b8c…`, cases `range(0,128,8)`, one worker, full prompt;
   - 2048 prefill, 0.25 GiB prefix cache;
   - image `natlang:pop-causal-conv1d-torch214-cu134-sm89-v1`, `--cpus 3 --memory 5g`.

   It serves the checkpoint's own runtime. Outputs go to `natlang-independent-recovery-20261031/pop-behavior-eval/`.
6. **Collection.** It copies `completed.json`, `launch.json` and a minimal detail (case id, family and outcome label
   only) back to the DGX. Protected packet content stays on Pop and never enters training.
7. **Runtime-matched parent.** `train/promotion.py` pairs arms only under an identical Python runtime. When the
   parent has no receipt under the candidate's runtime, the parent is evaluated once for that runtime and cached in
   `state.json` `parent_evals`.
8. **Comparison and selection.** It writes a `natlang.behavior-promotion/1` receipt and a
   `natlang.behavior-parent-selection/1` record. Promotion rule: `min_delta` 1, so the candidate must solve strictly
   more of the 16 cases (one-success flips are within this packet's repeat noise). A promoted candidate becomes the
   behavioural parent, and `behavioural-parent.json` is rewritten.
9. **EMA arm** (committed decision #2, register #11). When the checkpoint carries a weight EMA,
   `pop_eval_prepare.py ema` (40 GiB) writes the EMA as a model-only checkpoint plus an EMA view receipt and transfer
   manifest. It is served as a plain checkpoint and compared against the same parent receipt. An EMA export is
   evaluation evidence only and is never promoted to a training parent.

After each candidate, the watcher's own DGX views are pruned, except the parent's view and the peer's delta base.

## User profiles (#20)

`~/.config/natlang/config.json` `nativeSources.lfm-latest` now points to `W/ops/pop-behavior-eval/behavioural-parent.json`
(the `{checkpoint: {path, sha256, bytes}, global_step}` pointer format). `natlang setup --native lfm-latest --gguf q8_0`
pins it and refreshes `lfm-latest` and `lfm-latest-gguf`. On a promotion, every `natlang` run says the source has a
newer checkpoint. Rerun the same setup command (through the ledger) to move the profiles.

## Limits and failure handling

- A failing candidate is retried up to three times, then marked `failed` with `last_error`. The watcher keeps
  running. Unit logs are under `journalctl --user -u popeval-…` on the DGX or Pop.
- When the rolling slot already holds a newer step and no pin of the requested SHA exists, the expected-SHA check
  fails rather than evaluating the wrong weights.
- The comparison is 16 cases on one packet. It selects parents; it is not a broad capability claim and not a
  substitute for the decision gate.
