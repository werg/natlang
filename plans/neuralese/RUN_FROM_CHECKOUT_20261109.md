# Production RL runs from the checkout (owner, 2026-11-09)

Frozen runtime copies and code pins are gone from the RL path. The controller, collector workers, learner, DGX actor and
Pop actor run from `/home/werg/natlang`; a running process keeps its code and every newly started worker (one fresh
`run_bonsai_queue.py` per episode), learner (one process per update) or actor gets current main. Pop syncs with
`git pull --ff-only` (`pop_behavior_eval.sync_pop_checkout`) instead of receiving a runtime.

## What is recorded instead of checked (`common/code_provenance.py`: `{git_commit, dirty}`)

| record | field |
| --- | --- |
| round start (`round-N/generations.jsonl`, event `start`) | `code` (controller), `actor_code` per endpoint, `code_changes` vs the previous start |
| round result (`score.json`) | `code` |
| completed update (`state.completed_update_history[]`) | `code` (the checkout as of the learner launch) |
| actor attestation (`/v1/neuralese/info` `rl_policy_binding`) | `code` (and the Python inventory `python_runtime`, informational) |
| actor transport manifests | `code` |
| SGLang fork | `serving_engine.fork` = actual `commit`, `declared_commit`, `dirty` (warns on stderr when it differs from `sglang-fork.json`) |
| controller migration (`state.code_transitions[]`) | `natlang.rl-code-checkout-transition/1` |

Kept: actor policy SHA (`validate_actor_receipts`), native identity/tokenizer/embedding, prompt-rendering contract, schemas and
versions of every artifact. Removed refusals: see the commit message of "Production RL: code identity is recorded provenance".

## Moving w5 to run from the checkout

`bf95-lowlr-w5` (unit `bf95-lowlr-w5-rN`, out `.../ops/bf95-lowlr-1105/w5`) still launches from a frozen runtime. After the
tonight's last old-style cutover (or instead of it):

```
python3 scripts/neuralese_data/restart_controller.py /home/werg/data/lfm3-rl-dgx-20261029/ops/bf95-lowlr-1105/w5 --to-checkout --dry-run
python3 scripts/neuralese_data/restart_controller.py /home/werg/data/lfm3-rl-dgx-20261029/ops/bf95-lowlr-1105/w5 --to-checkout --at-round-boundary
```

`--at-round-boundary` waits (up to `--boundary-wait`, default 3 h) until the controller has finished its update and begins
the next round, then stops it cooperatively (zero redone episodes); without it the restart happens now and the in-flight
episodes are redone from their saved turns. After this restart the controller runs under the supervisor and later code changes
need only `restart_controller.py OUT --in-place`.

Changed by `--to-checkout` (each file keeps a `.pre-checkout-<time>` copy; a refused preflight restores them all):

* `config.json`: `actor_command[1]` -> `<checkout>/scripts/neuralese_data/serve_native_checkpoint.py`;
  `actor_environment` -> `{NATLANG_REPO: <checkout>, PYTHONPATH: <checkout>/training/neuralese:<checkout>}`;
  `runtime` -> `<checkout>/ts-host`; `after_actor_ready`: the `--runtime <frozen>` pair is dropped;
  `native_actor_python_runtime` and `native_actor_assets` are removed. `branch_origin` (provenance) and the hash-bound
  `task_eval.runtime` of the trainer parameters (dev-evaluation protocol; `task_eval_every` is 0) stay as recorded.
* `interleave-v49/control-w5/admit-learner.py`: the learner's `NATLANG_REPO`/`PYTHONPATH` runtime -> the checkout.
* `interleave-v49/env.sh`: `V=` -> the checkout (the shared env of prepare/next-window scripts).
* `launch.json`: payload `cd <checkout> && exec env NATLANG_REPO=<checkout> PYTHONPATH=... python controller_supervisor.py -- python -u -m
  natlang_neuralese.rl.production run --config ...`, `workdir` = the checkout (the ledger unit's `--workdir`).
* `state.json`: appends `code_transitions[]` (`natlang.rl-code-checkout-transition/1`: from runtime, to checkout + commit).

No other state record needs a transition for resume to accept the move: the open round's `collection-binding.json` keeps its
legacy `runtime`/`collector_sha256`/`python_runtime` fields and the comparison ignores them; the actor-engine transition
(`actor_engine_transitions`) is appended automatically with the new actor command path; `actor-prompt-binding.json`
still compares the prompt-rendering identity (a legitimate contract check, not code identity).

Not changed here: `recipe-runtime.json` (hash-bound copy; its `task_eval.runtime` names the dev-eval ts-host).

## Restart in place (workers and actor outlive the controller)

`restart_controller.py OUT --in-place` sends SIGUSR1 to the supervisor, which forwards it to the controller:

* collection: dispatch stops, each running collector worker is left running and recorded in
  `workers/worker-<tag>-detached.json`, the managed actor keeps running (`actor.pid.json`), the Pop actor is not stopped
  (`after_collection` does not run), the controller exits 75 and the supervisor starts it again (current main);
* update/SFT window: it finishes, then the controller exits 75 at the round boundary.

The relaunched controller adopts an actor that still attests the round's policy, lets each detached worker finish its own
episode on the code it started with (the worker's journal row is the result; its slot stays busy), dispatches only entries
nobody runs and re-dispatches an entry whose worker died unfinished. Workers and actor stay inside the controller's ledger unit
and claim, so no admission changes.
