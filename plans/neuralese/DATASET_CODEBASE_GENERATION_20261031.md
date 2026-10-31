# Dataset-backed deep codebase generation

Owner direction, 2026-10-31: resume supervision and up to six Luna generation
workers; generate genuinely diverse, deep Natlang codebases from diverse datasets,
then multiple distinct invocations against each. Renaming is not diversity.

## Why the previous data failed this requirement

`LFM3_DATA_COVERAGE_AUDIT_20261031.md` found that 89% of Big2 edges
carried identical instructional payloads and 89.9% of producer targets repeated
one 327-token program. Pair extraction repeated a chain's producer roughly
23 times. Different underlying records did not establish different programs.
The same audit identified sources silently lost between conversion, rendering,
registration and mixing. Large advertised row counts hid narrow actual training.

SFT step608 regressed in the aligned paired comparison (base 32/51 versus
trained 17/51). That is evidence of harm, not proof of one exclusive cause.
Retain diagnostic evidence; do not restart production from that parent. Local
RL from fresh base continues subject to actual progress and behavior checks.

## Authoring policy: application first

Luna authors each application as a natural-language design followed by typed
Natlang functions and callable folders. Dataset provenance and available facts
constrain that design; fabricated causal labels do not become dataset truth.

1. Select an acquired, permitted dataset and inspect its real schema, relationships,
   documents, workflows and existing split groups. Check the protection union.
2. Design a useful application around those properties. Write `DECOMPOSITION.md`:
   public operations, meaningful private subsystems, state, effects and exact
   service/oracle boundaries. Justify crisp code as mechanism, not hidden policy.
3. Implement a cohesive codebase with multiple operations and nontrivial dependencies.
   Depth means necessary cross-function reasoning and state/data dependencies,
   not padding files or calling trivial helpers. Preserve source revisions.
4. Author different invocations: investigate evidence, execute a workflow, handle
   a changed constraint, repair or extend behavior, reconcile state, recover from
   a partial failure. Select operations appropriate to the application; do not
   force the same set of wrappers onto every dataset.
5. Check expected outcomes independently of the reference implementation. Include
   meaningful wrong controls and alternative correct forms. Package loading is
   a separate check. Behavioral repair must pass behavioral tests, not merely load.
6. Execute actual teacher trajectories through the harness and retain full call
   graphs, effects, typed results, prompts, actor identity and verifier evidence.
   Reject incorrect attempts from positive SFT; retain failures for diagnosis and
   qualified preference pairs. Never fabricate model captures from oracle replay.
7. Convert verified complete graphs with current native-state custody and serving
   prompts. Admit useful additions into the repaired recipe and verify adoption.

Initial application candidates are event-log investigation and dependency/release
operations. These are authoring work in progress, not admitted data or capability
improvements. Dataset selection must be faithful; a convenient unrelated dataset
does not make a synthetic app dataset-backed.

## Diversity and independence

Track application/dataset/world identities separately from prompt variants and
invocation counts. Split the full related application/world component before
expanding invocations; train and held calls must not share protected knowledge.

Review computational problem, algorithm, call graph, information visibility,
state transitions, effects and failure recovery. Structural fingerprints and
normalized instruction/target repetition detect scaffolds; they do not certify
semantic diversity. Record unknowns and inspect actual examples. Count distinct
datasets, applications, operations and worlds alongside rows and own-output mass.
Include substantial behavioral tasks; structural source exploration is supplemental.

Measure the actual training mixture, including complete producer/consumer graphs,
and cap repeated templates and target bodies before expansion overwhelms new work.
Do not repeat a producer once per edge when one whole-chain graph is available.

## Execution and admission mechanism

Keep executable RL tasks separate from sampled, qualified SFT demonstrations.
An application can serve both. The recovered idea reservoirs, artifact states,
and feature-coverage gaps are tracked in
`PROGRAM_SUPPLY_AND_FEATURE_COVERAGE_20261031.md`; older useful work must be
crosswalked into this campaign rather than silently replaced or counted twice.

Reuse `run_reviewed_luna_dispatcher.py` (six slots maximum, one request per slot),
`run_bonsai_queue.py`, the current loader/verifiers, protection checks and native
graph converters. The master case list, runtime, source inventory and launch plan
are immutable and hash-bound. Keep authored development/confirmation tasks outside
the generation source. A fresh generation authorization does not qualify sources.

Run local jobs through the memory ledger with bounded scratch/cache writes. Keep
supervision in the primary strand; preserve productive RL and reserve enough
memory for its real peak. Luna generation uses hosted inference, not DGX GPU
capacity. Track actual active requests and accepted outputs, not just slot claims.

## Takeover observations

At initial inspection on 2026-10-31 local JOLT round6 completed its optimizer loop
through global step42 and state advanced to trained_rounds7. Its scheduled
checkpoint-input-offload measurement and runner handoff must be verified, not
assumed. SD round8 has collected data but previously failed updates. Pending
runtime cutover preflight subsequently returned zero for SD on runtime-d6ccac819.
No new RL capability lift is established by these observations.

Cloud autopilot logs still report `CWSANDBOX_GPU_NOT_ALLOWED` on cw1..cw4.
No GPU entitlement or resumed cloud training is claimed. Internal disk initially
had ~40 GiB free, external ~641 GiB; recheck before large writes.

Natlang's Luna credential had expired while the current authenticated Codex
session retained a valid same-account access token. The supervisor synchronized
access and actual expiry under the credential lock, preserving the prior private
credential and refresh policy. No secrets were logged. Verify a real request.

## Review findings during the first authoring pass

- LogHub produced a candidate application with seven public operations and 247
  source-derived calls. Static checks and scripted/helper execution are useful,
  but are not sampled teacher behavior. Its operations still share a template
  inference child; seven roots do not prove seven independent computational
  strategies or a sufficiently deep codebase.
- Its first exact tasks asked for dataset `EventTemplate` annotations from full
  raw log lines. Those annotations describe message content rather than all log
  headers, and wildcard placement need not be uniquely inferable from one line.
  Resolve this mismatch with faithful parsing/context and an appropriate verifier
  before mass collection. Do not train arbitrary annotation choices as the only
  correct behavior. Preserve the original candidate for diagnosis.
- The telecom worker initially authored a task-contract auditor. That may be
  useful supplemental data, but is not a telecom operating application. The
  primary application is being redirected to actual environment state and tools;
  source expected actions and acceptance criteria remain verifier-side.
- A third worker is authoring a relational application from a real acquired
  training database. These three lanes are authoring in progress, not three
  admitted sources or six active harness collectors.
- The initial diversity audit lost callable degree values and erased ordinary
  prose into identical tokens. Both issues were returned for correction and
  targeted tests. Structural fingerprints remain heuristics, never a semantic
  diversity certificate.
- Intake must preserve actual upstream protection groups and intentional package
  assets/notices. Different artifact-file hashes and aggregate provenance hashes
  must not be conflated. Loading, reference replay, live execution and admission
  are separate evidence stages.

The scheduled GPU measurement was accidentally stopped immediately after admission:
its caller mistook `memory_ledger.py run` returning for job completion. The
supervisor stopped that driver, cancelled the conflicting unstarted RL waiter and
started `codex-rl-gpu-window-2`. The dormant driver was corrected to wait for the
actual systemd unit's terminal state. Its `.done` suffix is not a success receipt.
Actual measurement results and the subsequent RL resumption still need checking.
