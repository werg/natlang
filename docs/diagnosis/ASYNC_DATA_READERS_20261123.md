# Data reader source routing and owner-only controller resumes

Inventory before the change (all paths are relative to the research checkout):

- `scripts/neuralese_data/expert_iteration.py`: Cycle already has `--data-repo`
  for mix builds and `--fill-repo` for fill generation, windows and Pop fills.
  Extraction, replay, own graph conversion and mixed-stage fill builds still use
  inconsistent code/environment routing. AsyncRoundDriver.cycle_args discards
  both choices, prepare_plan builds with the runtime pin, and _fill_beside launches
  its reader under runtime-pin provenance. The async CLI exposes neither choice.
- `sft_sources.py:candidate_turns` reloads an observed-live original manifest;
  `filled_reasoning.py:open_goal_of` and Renderer.chains reload it too. New native
  success plus rejected-review corrections require the completed-reassessment
  reader; copying an already rendered row does not upgrade those consumers.
- DGX archive `drain-archive.py` and `export-recovered-current.py` resolve Python
  modules from the selected immutable archive-code-current pin, subject to an
  explicit NATLANG_REPO. `deploy-semantic-recovered-pipeline.py` is an older
  deployment producer and must not reinstall an obsolete pin. The ingester poll
  and dedicated fill keeper each have their own selected source/environment.
- The standing async owner's global fill and data/mix jobs are separate consumers
  from that keeper. The observed owner was pin1c177e3fc609, with live WORK included
  explicitly, an active r0002 data job and an inactive global fill during SFT.
  These cannot be upgraded merely by editing the dedicated keeper config.
- `ops/ei/watch/semantic-next-round-handoff.py` is a one-time cycle-to-async
  transition; it explicitly refuses when phase-state.json exists. Its JSON is
  not an active async owner's next-round loader gate.
- The model runtime, every actor engine, held runtime and learner use the actor's
  existing --commit/--pin or already committed chains. This change routes data
  tooling, not inference, scoring, parameters or optimizer state. Native SDK
  decoder products remain the separate frozen Core a87 release. Colab SFT reads
  the committed chains via AsyncRoundDriver.colab_command/sft_env; it does not
  reload these original live manifests.

The async default now uses its own controller source for data/fill tooling, while
preserving its actor/learner --commit/--pin. Explicit --data-repo and --fill-repo
select immutable reader pins; supplying the runtime pin opts out. Cycle's fill
reader follows its selected data reader unless explicitly overridden. Every
source consumer's command and NATLANG_REPO/PYTHONPATH agree. The controller records
actual controller/runtime/data/fill Git identities separately in reader-sources.json
and in its dry-run plan, never claiming that new data tooling changed actor code.
--minimum-data-reader-commit binds both immutable data/fill readers at controller
startup and every SFT boundary; it deliberately does not demand a new actor or
learner pin. This is the active owner's loader gate, replacing reliance on an
already-completed one-time handoff. Without that explicit source requirement,
legacy-pin paths remain an opt-out for data that does not need newer readers.

An owner-only resume stops/cancels only the old controller unit, then starts the
new controller with the original root, runtime arguments and policy recipe plus
explicit reader paths. Independent actor/learner/data units survive SIGTERM.
Prepared stages and phase-state are reused. A running/requeued data job's existing
script is preserved, joined and recorded complete before a new build; a resume
must not overwrite a script an old child is still reading. The operational
publication gate must also wait for any old-source live readers still running.
Only then is a pending recovered receipt published and ingested. Current learner
chains, optimizer, RNG, policy and checkpoint are never rewritten by this routing.

Targeted coverage includes actual CLI/argv routing, a distinct fill-reader
execution environment, explicit opt-out, preserved active data scripts, and
SIGTERM of the actual async run loop followed by phase resume while an independent
child and optimizer/RNG/checkpoint fixture bytes remain intact. Operational proof
records real child PIDs/phase/policy and selected pin identities before and after.

A live follow-up found two admitted and four ledger-requeued Vibe fill clients:
the keeper treated preemption as termination and launched timestamped replacements.
The Vibe keeper now queries the authoritative ledger's waiting_order, preserving
requeue ownership, and exactly five duplicate clients were retired while the
oldest admitted worker and all actor/learner/server units remained intact.

The same lifecycle inventory includes expert_iteration.ledger_queued,
grpo.unit_active (detached Pop eval), and static_curriculum_campaign.render and
ensure_teacher_queued. They now use a shared read-only ledger ownership query,
including admitted claims across the stop-to-requeue transition. It performs no
GPU probe and changes no priority/backfill/reclaim semantics. Unknown/corrupt
ledger state raises instead of pretending a job ended. target_set_watcher.ledger_knows
already fails closed on unknown state (lines93–101), so needs no correction.
Dashboard/watchdog full status calls need actual memory telemetry and remain.
