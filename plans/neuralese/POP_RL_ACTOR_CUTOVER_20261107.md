# Pop as a second RL rollout actor: cutover note (2026-11-07)

Built and tested; not cut over. The live controller (`bf95-lowlr-w3`, frozen runtime v57) is untouched.

## What exists

| piece | where |
| --- | --- |
| remote endpoints in the controller | `rl/production.py`: config `remote_endpoints`, `remote_workers`, `remote_actor_ready_seconds`, hook `after_collection`; `round_endpoints`, `round_remote_workers`, `worker_slots`, `weighted_split` |
| Pop actor lifecycle (hooks) | `scripts/neuralese_data/pop_rl_actor.py start / stop / status` |
| Pop container launch | `scripts/neuralese_data/pop_dev16_driver.py`, job `mode: rl-actor` (same image, limits and serving flags as the dev16 eval) |
| view prepare / send / runtime install | `pop_behavior_eval.py` (`prepare('rl', ...)`, `send`, `ensure_pop_runtime`, `make_pop_room`) and `pop_eval_prepare.py rl` |
| watcher respects the RL actor | `pop_behavior_eval.py`: `rl-actor-active.json` marker; `evaluate()` waits while it exists (marker older than 6 h is stale) |
| RL transport scope, cross-device export | `native_bundle.TRANSPORT_SCOPES` (`rl-controller-pin`), `serve_native_checkpoint.py --actor-export-format`, `Engine.set_actor_export` |

Why the serving changes were needed (not optional):
- `validate_actor_receipts` refuses a `diagnostic-local-pin` view as an RL policy, and `independent-recovery` needs a private
  cloud export receipt of every round's checkpoint. `rl-controller-pin` is the same local-pin verification (tensor equality
  against the source at prepare time, source SHA bound in the transport) for the controller's own policy, with no recovery claim.
- Tokens sampled by an actor of the *native* engine go through the learner's exact pre-SGD parity gate (`atol=1e-4`,
  `native_replay.py`); one Pop episode as the first native episode would abort the update. Pop's log-probs differ from a DGX
  native actor's by mean |d| 0.006-0.022 and max 0.10-0.17 per token (measured, below), so the Pop actor binds
  `actor_export` {kind exact, format `native-pop-rtx4060-bf16`}. The learner then recomputes its tokens, applies the per-episode
  engine guard (`max_abs_logp 1.5`, `mean_k3 0.01`) and the loss's TIS correction, exactly like the live SGLang actor.

## Config for w3 (keys and values)

Add to `ops/bf95-lowlr-1105/w3/config.json` (all other keys unchanged; `endpoints` stays `["http://127.0.0.1:8200"]`, `workers` stays 4):

```json
"remote_endpoints": ["http://100.73.236.5:18531"],
"remote_workers": {"http://100.73.236.5:18531": {"workers": 3, "weight": 0.4}},
"remote_actor_ready_seconds": 300,
"after_actor_ready": ["/home/werg/natlang/.venv-neuralese/bin/python", "<NEWRT>/scripts/neuralese_data/pop_rl_actor.py", "start",
                      "--checkpoint", "{checkpoint}", "--policy-id", "{policy_id}", "--runtime", "<NEWRT>"],
"after_collection": ["/home/werg/natlang/.venv-neuralese/bin/python", "<NEWRT>/scripts/neuralese_data/pop_rl_actor.py", "stop"]
```

`<NEWRT>` is the new frozen runtime. The Pop address is the tailscale IP `ssh -G pop-os` resolves to (the actor binds exactly it;
the DGX reaches it, measured). `remote_workers` is required for every remote: `workers` run on the managed endpoint (weight 1 each);
each remote adds its own workers, and a worker's weight is its relative share of entries (`weighted_split`, deterministic). With
the numbers above Pop takes about 3x0.4 / (4 + 3x0.4) = 23% of the entries on 3 concurrent requests. Endpoints and `remote_workers` are
transport, not provenance (2026-11-07, plans/neuralese/CONTROLLER_RESTART.md): the collection binding no longer records them,
each (re)start appends its actual endpoints and actor receipts to `round-N/generations.jsonl`, and a restart may add or remove
Pop, or change worker counts, at any time.

Every (re)start of a round's collection: the remote is included only if it attests the round's policy, native identity, Python
runtime, prompt rendering within `remote_actor_ready_seconds` after `after_actor_ready` (full `validate_actor_receipts` against
the managed actor); otherwise the round continues on the DGX alone (logged as `remote RL actor ... usable: false`). Entries that
earlier starts finished are never redone; pending ones are split over whatever endpoints are usable now. The hooks never fail
the round: `start` exits 0 when Pop is unreachable, busy or unattested; `after_collection`
runs in the controller's `finally` after the DGX actor has stopped and a failure is only logged.

Optional hook flag `--defer-to-eval`: skip Pop for the round while the watcher has a queued, running or undiscovered candidate.
See risk 1.

## Cutover steps (supervisor)

1. Freeze a runtime from the main commit that contains these changes: `scripts/neuralese_data/freeze_native_runtime.py --commit <sha>
   --predecessor runtime-native-corrected-mix-v57-1106 --out runtime-native-corrected-mix-vNN-1107 --control-bindings
   ops/bf95-lowlr-1105/interleave-v49/v50-bindings.json` (git archive of the commit, Node closure, protected inputs, file pool seal;
   what produced v50-v57, logs in `interleave-v49/freeze-vNN.log`).
2. Rebind and restart at any time (no round boundary needed any more): `scripts/neuralese_data/restart_controller.py
   ops/bf95-lowlr-1105/w3 --runtime-from-main` does the freeze, cooperative stop, `rebind-runtime.sh`, preflight and relaunch in one
   command (plans/neuralese/CONTROLLER_RESTART.md). Add the config keys above before running it (or `--freeze-only` first).
3. Restart `pop-behavior-eval` (new unit name through the ledger) so the watcher loads `wait_for_rl_actor`; its state is persistent.
4. The first round after the cutover is the real qualification: check `controller.log` for `remote RL actor ... usable`,
   `collection-binding.json` endpoints, `actor-prompt-binding.json`, per-worker journals, the Pop container (`docker ps`
   shows `poprl-<sha12>`), and the learner's `engine_guard` summary (rejected episodes, `max_abs`, `max_mean_k3`).

## Measurements (2026-11-07, w3 step-98 policy a854332c, scratch runtime = v57 + these files, all from the DGX)

- Cold start of the hook (view prepared and cached, delta/already-verified send, container start and attestation): 33 s; first
  time including the 70 s view preparation and runtime install: 127 s. Teardown: container removed, GPU back to 2 MiB.
- Pop actor alone, 4 concurrent requests (4 workers, real w3 tasks, full prompt, top-k 128, context 32768): decode 54-58 tok/s
  aggregate (about 14 tok/s per stream), prefill about 360-620 tok/s (turn prompts are 4k+ tokens). The DGX SGLang actor
  measured at the same time: about 167 tok/s. Pop is therefore about a third of the DGX actor.
- Pop GPU memory 6.1-7.1 GiB used of 8.0 (the actor caps itself at 90%); container host memory at most 2.8 GiB. The 4 GiB
  container cap (the RL claim; the eval uses 5) held without OOM kill for a 6-minute run at 4 concurrent requests.
- Concurrency sustained: 4 active sequences (all 4 workers), no queueing. Two real collections completed through
  `collect()` against Pop alone (4 rollouts each, `collection_complete`, actor receipt with `actor_view.scope
  rl-controller-pin` and `actor_export`); 3 of 4 rollouts of each hit the shortened case wall (900 s / 360 s), expected at
  this speed.
- Two-endpoint run through `collect_with_actor` (managed DGX native actor on the same view + Pop, `remote_workers`
  {workers 1, weight 1}, hooks start/stop): hook reused the already-serving Pop actor, the remote attested through the full
  `validate_actor_receipts`, 3 workers (queue split 2/1/1, endpoints 8201/8201/Pop), receipts of both endpoints in
  `score.json`, `after_collection` stopped Pop, collection complete.
- Blocks: the collected episodes used no Neuralese state, so no block bundle was produced. Fetching a stored block over HTTP
  from Pop works: 961-token block, 143 MiB, 15 MiB/s.
- Pop vs DGX native actor log-probs (same view, same deterministic seed, temperature 1, 3 prompts x 200 tokens, while the
  sampled prefix agrees: 129/200, 200/200, 80/200 tokens): mean |d logp| 0.011 / 0.006 / 0.022, max 0.17 / 0.10 / 0.17,
  mean k3 4e-4 / 1.5e-4 / 8.5e-4. That is 100x the learner's exact gate (1e-4) and well inside the engine guard
  (max 1.5, mean k3 0.01): the reason for the `actor_export` binding. This compares two actors, not a learner recompute (the
  DGX native actor and the learner share arithmetic; the SGLang path has the same guard). No cheaper learner-side probe exists
  (the guard runs inside the update; `NATLANG_ENGINE_GUARD_LOG` records it per episode).

## Risks and open items

1. Evaluation vs rollouts: every update boundary produces a checkpoint that the watcher evaluates (~55 min on Pop) just as the
   next collection starts. With rollouts taking priority (as requested) the dev16 gate result of an update can be delayed by the
   whole next collection round (about 5 h); with `--defer-to-eval` the round goes without Pop whenever an evaluation is pending
   (usually). Supervisor decision.
2. Pop is about a third of the DGX actor (decode 50-56 tok/s at 4 concurrent requests against about 167 tok/s on the DGX SGLang
   actor now), so expect about +20-30% rollouts per hour, not 2x. The weights above are a guess from that ratio; check the
   per-worker journals after the first round and adjust `remote_workers` (it applies to the next unbound round).
3. Pop GPU memory reached 7.1 GiB used of 8.0 (the actor caps itself at 90%) with 4 concurrent requests; long contexts at
   4 concurrent may OOM a request (it becomes an infrastructure error outcome, like other transient failures). Hence 3 workers.
   Pop host memory is tight: `available` was 5.3 GiB and falling (unreclaimable kernel slab 3.9 GiB, vmalloc 3.7 GiB), the eval's
   5 + 0.5 GiB ledger claim was refused at 5.4 free; the RL actor claims 4 + 0.5 (container cap 4g) and the next evals may need
   the same reduction or a Pop reboot.
4. Neuralese blocks of Pop-sampled episodes are fetched over HTTP at the end of collection: a 961-token block is 143 MiB and
   fetched at 15 MiB/s from the DGX (wifi + tailscale). Round 0 of w3 had 4.5 GB of blocks for 144 rollouts (3% Neuralese-using),
   so Pop's share is well under 1 GB per round.
5. Pop disk: 19 GB free at 98% use; each view is 3.4 GB (an exact delta of 0.3 GB when the previous view is the base);
   `make_pop_room` prunes older derived views.
6. The Pop actor serves the policy from a model-only view with scope `rl-controller-pin` (no independent recovery claim). The
   training checkpoint, not the view, remains the policy's custody.
7. Not run: a real two-endpoint round in the live controller, or the learner consuming Pop-sampled captures (the engine-guard
   path is shared with SGLang captures; its first real exercise is the first production round).
