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
the numbers above Pop takes about 3x0.4 / (4 + 3x0.4) = 23% of the entries on 3 concurrent requests. `remote_workers` is bound
into the collection binding together with the endpoint list (only when a remote joined); a round without a remote binds exactly
as before.

Fresh round: the remote is included only if it attests the round's policy, native identity, Python runtime, prompt rendering within
`remote_actor_ready_seconds` after `after_actor_ready` (full `validate_actor_receipts` against the managed actor); otherwise the
round collects on the DGX alone (logged as `remote RL actor ... usable: false`). Bound round (resume): exactly the recorded
endpoints and `remote_workers`; a bound remote that cannot be brought up fails the resume with a clear error (it never collects
on another set). The hooks never fail the round: `start` exits 0 when Pop is unreachable, busy or unattested; `after_collection`
runs in the controller's `finally` after the DGX actor has stopped and a failure is only logged.

Optional hook flag `--defer-to-eval`: skip Pop for the round while the watcher has a queued, running or undiscovered candidate.
See risk 1.

## Cutover steps (supervisor)

1. Freeze a runtime from the main commit that contains these changes: `scripts/neuralese_data/freeze_native_runtime.py --commit <sha>
   --predecessor runtime-native-corrected-mix-v57-1106 --out runtime-native-corrected-mix-vNN-1107 --control-bindings
   ops/bf95-lowlr-1105/interleave-v49/v50-bindings.json` (git archive of the commit, Node closure, protected inputs, file pool seal;
   what produced v50-v57, logs in `interleave-v49/freeze-vNN.log`).
2. Rebind w3 between rounds (the collection binding pins the actor Python inventory, which changes with any package change):
   `interleave-v49/rebind-runtime.sh w3 v57-1106 vNN-1107` after the in-flight round's collection completes. It rewrites the config
   paths, `native_actor_python_runtime`, `native_actor_assets`, `actor_environment`, the learner admission wrapper and `env.sh`. The
   controller and learner then run vNN. Add the keys above in the same edit.
3. Restart `pop-behavior-eval` (new unit name through the ledger) so the watcher loads `wait_for_rl_actor`; its state is persistent.
4. The first round after the cutover is the real qualification: check `controller.log` for `remote RL actor ... usable`,
   `collection-binding.json` endpoints, `actor-prompt-binding.json`, per-worker journals, the Pop container (`docker ps`
   shows `poprl-<sha12>`), and the learner's `engine_guard` summary (rejected episodes, `max_abs`, `max_mean_k3`).

## Measurements (2026-11-07, w3 step-98 policy a854332c, scratch runtime = v57 + these files)

See the report that accompanied this note; numbers are repeated here once the runs finished (below).

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
4. Neuralese blocks of Pop-sampled episodes are fetched over HTTP at the end of collection: a 961-token block is 143 MiB and
   fetched at 15 MiB/s from the DGX (wifi + tailscale). Round 0 of w3 had 4.5 GB of blocks for 144 rollouts (3% Neuralese-using),
   so Pop's share is well under 1 GB per round.
5. Pop disk: 19 GB free at 98% use; each view is 3.4 GB (an exact delta of 0.3 GB when the previous view is the base);
   `make_pop_room` prunes older derived views.
6. The Pop actor serves the policy from a model-only view with scope `rl-controller-pin` (no independent recovery claim). The
   training checkpoint, not the view, remains the policy's custody.
7. Not run: a real two-endpoint round in the live controller, or the learner consuming Pop-sampled captures (the engine-guard
   path is shared with SGLang captures; its first real exercise is the first production round).
