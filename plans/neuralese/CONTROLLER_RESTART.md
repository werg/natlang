# Restarting the live RL controller at any time (2026-11-07)

Owner requirement: "I want us to be nimble": change code, add or remove actor endpoints, change worker counts, and restart
the interleaved RL controller at ANY time, without waiting for a round or window boundary.

## Commands (2026-11-09: production runs from the checkout)

```
# restart the controller inside its unit; collector workers and the managed actor keep running, the relaunched controller
# (newest main) re-attaches. During collection it detaches at once; during an update/SFT window it finishes first.
python3 scripts/neuralese_data/restart_controller.py OUT --in-place
# one-time move of a controller that still runs from a frozen runtime (stop, rewrite config/wrapper/env.sh/launch.json, preflight, relaunch)
python3 scripts/neuralese_data/restart_controller.py OUT --to-checkout [--at-round-boundary] [--dry-run]
# plain stop/relaunch (config edits: endpoints, workers, after_collection, ...)
python3 scripts/neuralese_data/restart_controller.py OUT
```

`--in-place` needs the controller under `scripts/neuralese_data/controller_supervisor.py` (the unit's main process; it
relaunches the controller on exit 75 and forwards SIGTERM/SIGINT/SIGUSR1). `--to-checkout` installs it. Plain restart flags:
`--grace S` (default 600) / `--patient`, `--no-preflight`, `--dry-run`, `--unit`. The frozen-runtime flags and
`freeze_native_runtime.py` are deleted; see `RUN_FROM_CHECKOUT_20261109.md`.

Plain restart, in order: (1) cooperative stop: SIGTERM to the controller only; it stops its workers and actor and archives
retained partials; a learner/SFT trainer finishes its update and checkpoints; after `--grace` the unit is stopped hard;
(2) relaunch through `memory_ledger.py run` as `<base>-rN` from `launch.json`; (3) wait until the new controller holds
`controller.lock` and has rewritten `state.json`.

## What makes a restart seamless (code on main, commits below)

* `collect()` no longer binds `endpoints` or `remote_workers` (transport, not provenance; each endpoint's policy identity is
  authenticated by `actor_receipts()` at every start). Old bindings that contain them still resume (compare ignores them).
* Collector/actor **code identity** is provenance, never a gate: bindings carry no dist hash map, collector hash or Python
  inventory; each start records the git commit and dirty flag of the controller and every actor (`generations.jsonl`
  `code`, `code_changes`; `score.json` `code`; the completed-update record `code`).
* Every start of a round's collection is a *generation*, appended to `round-N/generations.jsonl` (endpoints, worker count,
  slots, actor receipts, accepted code changes, per-phase pending counts). Workers write new journal files per generation
  (`workers/journal-g2-0.jsonl`, `journal-g2-teacher-0.jsonl`); nothing is overwritten. Entries with a `finish` row in ANY
  journal (all generations, student and teacher) are never dispatched again; the pending ones are split over the current
  slots (weighted by `remote_workers`). `journal.jsonl` merges every generation. A rollout keeps the endpoint that sampled
  it (`rl_endpoint` restored from the generation that finished it).
* The JOLT+ teacher plan (a model call) is persisted in `round-N/teacher-plan.json` before the first teacher starts, so a restart
  reuses the exact keys instead of re-planning.
* An entry interrupted by the stop has a `start` row and no `finish` row, so it is retried; its partial job (journaled model
  responses, keyed by entry not by worker) replays on whatever endpoint takes it. `run_bonsai_queue.py` refuses to record a
  `finish` once SIGTERM arrived, so a stop that reaches the node child first cannot turn into a failed attempt.
* A memetic round whose after-collection work (distillation, ablation, held promotion) was interrupted is finished on the next
  start (the sealed collection returns immediately; the actor lease is taken for the after work only). Promotion is idempotent
  per parent.

## Per-phase behaviour on restart

"Stop" = cooperative SIGTERM to the controller. "Kill" = SIGKILL/OOM/power loss of everything.

| phase | stop (restart script) | kill | cost on resume |
| --- | --- | --- | --- |
| select tasks / sealed selection | selection snapshot is durable; same selection reused | same | none |
| student collection | workers stopped (seconds), partial jobs kept, rollouts scored so far, blocks archived; `collection_complete: false` | same minus the final archive (blocks stay in the actor store) | the in-flight model request of each running episode (about a minute) plus, if the code identity changed, partials whose provenance changed restart from scratch; finished entries never rerun. Before: a changed endpoint list or any code change refused the resume |
| JOLT+ teacher plan | plan persisted once | same | none (before: the NL plan could differ and orphan started keys) |
| teacher collection | as student collection, with `teacher-` journals | same | as above |
| scoring / block bundle | rerun from journals (deterministic, minutes for the block fetch) | rerun | minutes of fetch |
| memetic after-collection | finishes on next start if the stop interrupted it | same | the interrupted sub-step (distill/ablate/promote) reruns; promote reuses its recorded entry |
| learner update (`native_stage`) | trainer finishes the current update (up to about one update, 35-45 min) and checkpoints at the boundary; with the grace escalation the unit is stopped and the rolling checkpoint resumes | rolling checkpoint (commit 89fb6ba45) | at most the in-flight update (about 35-45 min of GPU) if the stop is escalated; none if the boundary was reached; capacity proofs re-measure when the learner code identity changed (a few minutes) |
| SFT window (`interleave.active_window`) | `execute` stops at an update boundary, `rolling_sft_resume` continues; window config keeps its recorded runtime paths (a code change reaches the next window) | same | at most one update |
| publication (`after_update`, checkpoint write) | atomic; the controller state advances only after publication | same | none (checkpoint replacement is atomic, the previous is preserved) |

## Still not seamless

* A learner or SFT update in flight is not interruptible mid-update: a restart waits for its boundary (up to about 45 min with
  `--patient`) or redoes that update after the grace (the default). Cutting an update short without losing its gradient
  work needs trainer support for micro-batch-level save, which does not exist.
* A running SFT window keeps its recorded runtime paths: code changes reach the next window.
* After a crash (kill), blocks of finished Neuralese-using episodes that were sampled on an endpoint you then remove are only in
  that endpoint's store; a cooperative stop archives them first. Keep the endpoint until the round is bundled, or restart
  cooperatively.
* Mixed-code rounds: episodes of one round can be sampled by different actor/collector code identities (recorded per
  generation). The learner's engine guard (and the exact parity gate for native actors) still checks every sampled token under
  the learner's own code before any update.
* A bound-`strict_code_identity` or a change in cases, policy, sampling parameters (`rows`, `group`, seeds, max turns, prompt
  rendering, context tokens) still refuses the resume, deliberately.
