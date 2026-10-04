# Projection sampling after direct SFT

## Decision — 2026-10-03

User clarified that the objective is the paper's projection-sampling approach,
including iterative expert-guided proposals and ordinary-prompt student likelihood
selection, followed by SFT. The earlier rewrite-only-first decision was a mistaken
interpretation and is superseded. Simplicity is not permission to omit the search.
Keep ordinary completion-only SFT and Muon. Existing rewrite-only code is reusable
proposal/execution infrastructure, not the intended experiment or a reproduction.
No rewrite-only collection has launched or entered training.

Sources studied: [paper](https://arxiv.org/html/2610.02140v1),
[project](https://aakaran.github.io/finetuning_with_sampling/),
[official implementation](https://github.com/aakaran/finetuning-with-sampling/tree/6d3e9f0bfaa98dcca534247dd35dc1b33dd8c428).
Saved research inputs: `runs/finetuning-with-sampling-research-20261003/`.

The paper conditions proposals on expert information and scores them under the
ordinary student prompt, then performs SFT. Its theoretical algorithm includes
proposal-corrected MH acceptance. The released implementation instead accepts
higher mean-token likelihood candidates greedily; it grades completed outputs
but also writes incorrect ones. The paper includes a rewrite-only SFT baseline:
rewriting alone is not established as consistently superior to ordinary SFT.
The implemented lane is that baseline, not the now-authorized target. Extend it
with search before launching the intended experiment. Distinguish the released
greedy search from theoretical proposal-corrected MH in code, receipts and reports;
do not silently substitute either or claim MH guarantees for greedy selection.
Their reported models are3B/7B, not our350M; applicability is an experiment.

## Existing proposal infrastructure (not the complete target method)

`ts-host/scripts/rewrite-student-trajectories.mjs PLAN` runs a no-provider preflight.
`--execute --sha256 PLAN_SHA` collects under an exact root-reviewed plan. It:

- Pins the frozen runtime, selected train IR, full source closure, and parent
  teacher artifacts. Rechecks source holds, native materialization, and curriculum
  admission. Refuses silently truncated references and duplicate programs.
- Runs the actual student on the original task in a fresh runtime. A teacher action
  sequence is appended only to the provider's proposal context. No reference is
  inserted into the program IR, runtime observations, or stored SFT context.
- Records privileged requests separately in proposal sidecars. Successful outputs
  remain supervision; do not try to remove correct answers from target actions.
- Gives one complete episode attempt, then a second only after failure. Defaults
  in the64-case candidate:8turns,32requests,512output tokens,temperature0.6.
  These are collection limits, not language deadlines.
- Uses `executeProgram` for whole-episode return/effects/files/world checks, then
  `admitRow` and `materializeNativeRows`. Alternative valid actions are allowed.
  No tool observation is copied from the teacher after a student action changes.
- Retains all raw failures. Interrupted/transport/resource failures do not become
  DPO negatives. Semantic failures can be considered later only by existing
  reviewed failure-pair rules. Exhausted failures add no positive examples.
- Emits `admitted.jsonl`, `turns.jsonl`, and `summary.json`. None automatically
  publishes or replaces the current training corpus.

`create_student_rewrite_pipeline.py` creates a3-stage config for our existing
`run_training_pipeline.py`: collect → ordinary chat-template render → token audit.
The student server must already serve the immutable reviewed checkpoint. The
collector checks its loaded-checkpoint identity (base revision and adapter
file hashes captured at startup), model ID and checkpoint pins. It never starts another model
server or takes GPU memory without an explicit experiment launch.

The CPU/GPU student server now honors explicit finite temperature and seed;
default remains greedy. Active sealed evaluation servers have not been modified
or restarted. Request-scoped RNG isolation applies only when a seed is provided. New student
servers default to loopback; container deployments must use host networking or
explicit container bind plus a loopback-only published port.

## Prepared candidate, not an active experiment

`runs/student-rewrite-20261003/rewrite-plan-v6.draft.json` selects64 short admitted
Qwen v9-v3 workflow episodes from its protected, source-reviewed train selection.
Its early step5000 adapter is a compatibility placeholder, not our late-SFT
student choice. Plan remains unapproved. `pipeline-v6.draft.json` is its derived
stage graph. Preflight passed64/64 with zero provider requests; no new unit tests
or model experiments were run for this implementation.

Do not change a reviewed plan after launch. Choose a versioned output/plan for
later weights. Normal SIGTERM aborts collection and writes interruption evidence;
continuing an interrupted attempt requires review and a versioned continuation.
An orphan proposal sidecar without its receipt blocks replay rather than being
overwritten. An unclean kill can leave `collector.lock`; verify the old collector is absent
before recovering it. This lane does not yet offer token-level replay resumption.

## Apply after substantial SFT

1. Continue the full main SFT epoch. Use a stable late-SFT snapshot (prefer the
   completed epoch for the first phase boundary). Preserve a complete immutable
   parent checkpoint including Muon buffers, scheduler, RNG and corpus cursor.
2. Implement and launch projection search against that snapshot: expert-guided
   suffix proposals, ordinary-prompt likelihood scoring, iterative acceptance,
   and task-equivalence checks. Retain the valid original teacher trajectory as
   the initial/reference candidate. Preserve useful admitted results as training
   data; do not launch the rewrite-only baseline as the requested method.
3. Review yield and behavioral shortcuts. A higher acceptance rate alone is not
   sufficient: compare task success on the unchanged protected evaluation packet,
   protocol failures, output length, and retention by task family. Never use
   heldout teacher answers in proposal generation. Later expand the cohort across
   train families and preserve the25% file/directory reducer mix target.
4. Render and token-audit admitted turns with the same tokenizer. Preserve original
   source IDs/groups and use new trajectory/turn IDs. Add an explicit lineage
   entry; keep rewrites distinct from new-source coverage.
5. Use reviewed `prepare_training_append_intake.py` and `--append-manifest` to
   continue a copied complete checkpoint; do not use `--init-adapter`, which resets
   optimizer state. Existing exclusions and a chained append need a reconciled
   transition before live use: `--append-manifest` and `--exclusion-manifest` are
   currently mutually exclusive, and the main run is an exclusion-resumed child.
   This is the principal remaining training handoff implementation issue.
6. Start with a bounded extra SFT exposure on the admitted lane. Keep fixed heldout
   evals, W&B, checkpointing and emergency resume. Compare before/after the phase
   against the same snapshot's scores. A later cohort may refresh the reference
   student, but freeze one checkpoint throughout each collection round.

## Required search implementation

- Add a pinned student scorer under ordinary task contexts without teacher hints.
  Keep target-policy log probabilities separate from guided proposal probabilities.
- Implement the paper's blockwise suffix-resampling search. Record cut positions,
  sampled token IDs, proposal/target scores, random draws, rejection reasons and
  immutable model identity. Tune compute budgets explicitly, not by dropping search.
- Resolve and document the paper/code acceptance discrepancy: theoretical MH uses
  summed log probabilities and forward/reverse proposal terms; released code uses
  greedy mean-token likelihood improvement. Reproducing the released experiment
  and implementing the theoretical sampler are distinct modes, not interchangeable.
- For interactive episodes, altering an action requires fresh execution of its
  downstream observations and actions. Never splice changed actions into an old
  observation trace. Adaptation to tool episodes is additional methodological work;
  do not claim exact paper reproduction for it without describing that adaptation.
- Preserve existing source, task-equivalence, admission, tokenizer and heldout gates.
  Verification alone does not replace likelihood-based search.

## Online tradeoffs — discussion, not an activated training change

The paper searches against fixed reference weights before SFT. An online extension
may search against a recent student snapshot, train on accepted candidates, then
refresh. Freeze weights for an entire search chain: changing weights mid-chain
changes its target and invalidates ordinary fixed-target MH reasoning. Old scores
must be recomputed under new weights before continuing a refreshed chain; old
KV caches cannot be carried across weight updates.

Three choices: (1) one offline search corpus, (2) bounded search/train rounds,
(3) continuously fed asynchronous search with explicit maximum checkpoint lag.
Rounds are the proposed starting point, not a user-approved final choice. Fully
synchronous per-minibatch search is freshest but can stall training for expensive
sampling and runtime verification. Async search overlaps work but introduces stale
weights, GPU contention and model-version/cache coordination.

Online timing alone does not remove a training forward pass. Guided proposal
contexts differ from unguided scoring/training contexts. In principle the final
unguided scoring forward could retain its graph and serve as the SFT forward if
weights, loss masks and tokenization match; retaining graphs for candidates while
search/verification proceeds has memory and batching costs. Measure total tokens,
GPU time, memory and heldout gains; do not assume this saves wall time.

Remaining checkpoint handoff work: support the main run's existing exclusion
lineage plus chained corpus transitions, permit a new phase after the old order is
exhausted, and explicitly start a positive next-phase learning-rate schedule while
preserving Muon buffers, RNG, weights and consumed-example provenance. Inheriting
an exhausted cosine schedule's zero learning rate would not train the new phase.

## 2026-10-04: completed crisp SFT and interactive projection implementation

The main epoch completed cleanly at global step13164,105311 trained examples,
zero overlength skips. This is an example-target completion; the old nominal
13165-step horizon is not the authoritative completion condition. Parent weights,
Muon/AdamW buffers, scheduler and RNG remain intact.

`project-student-trajectories.mjs` adds request-boundary suffix MH for interactive
programs. This is an adaptation, not an exact token-block reproduction of the
paper/released greedy sampler. It uses summed ordinary-context student token log
probabilities, temperature-only guided forward/reverse probabilities including
assistant EOS, and a fixed state-independent cut set. Whole changed suffixes run
fresh; prefix observations must match. Invalid tasks reject; model/transport errors
remain distinct. Concurrent model calls/external service/world state are held.
All proposals, token IDs/logprobs, cuts, draws and rejected outcomes are retained.
Only searched, changed, execution-admitted chains with accepted moves become
candidate native turns. No convergence guarantee or automatic training admission.

The existing protected-train64-case reference packet is entirely customer-service
reducers, not a broad post-training corpus. First corrected run uses four cases;
expand to balanced semantic/computational/effects/repair families with separate
source closure before a meaningful post-training phase. Never use the protected
execution evaluation cases or their answers for training.

A real serving bug was found: tokenizer(rendered_prompt) added a BOS to a chat
template already containing BOS. Training's frozen encoder correctly disabled
special-token addition. Corrected serving uses `tokenize_chat_prompt`, with a
regression check. Old periodic CPU execution results and final GPUv1 are retained
as duplicate-BOS measurements, not deployment-quality results. Projectionv1 was
interrupted and explicitly invalidated; no training publication. Corrected paired
base/final GPU evaluationv3 runs before projectionv2; evalv2 failed before requests
because its /select URL still assumed an endpoint ending in /v1.

New `posttraining_phase.py` and trainer `--phase-manifest` support a completed
parent (including exclusion lineage) followed by a new explicit positive-LR cosine
phase. Exact copied optimizer/weights/RNG are required; old counters/lineage remain,
new phase cursor starts0, global step/examples continue. Protected held-out row
contents and membership must remain exact. Token/mix/inventory audits and pinned
native/source gate receipts remain mandatory. Parent checkpoint is never edited.
Phase handoff has targeted synthetic checks; live post-training has not launched.

## 2026-10-04: diverse student projection and collector corrections

The corrected native-template base/final GPU packet completed: base 1/23 and
full-SFT student 11/23 successful executions, plus one held source in each arm.
This is a small diagnostic packet, not whole-system accuracy. The original
published LFM template token IDs match the training encoder exactly; training
used one BOS. The previous duplicate BOS was a serving defect.

Projection v2 completed four customer-service reducer cases, with three native
candidate trajectories / ten turns; none automatically entered training. Its
mostly constant-false labels are insufficient for the planned post-training mix.
A broader train-only reference bank now contains 128 current-admitted cases
across 39 families, selected from the pinned historical native trajectory bank.
All 128 are present in the completed parent's training membership and their
pinned prepared-teacher source groups map explicitly to train. Full protected
identity alias exclusion remains mandatory. Qasper's three pending-equivalence
references were held. Selection reports preceding that hold contain 131/40;
use the reviewed IR for current counts. No evaluation answers are reused.

Course changes / probability contract:
- Teacher initialization now executes its actions against fresh current contexts
  and current runtime contracts, then checks the entire outcome. Old system
  prompts/tool descriptions need not match. Retired actions are not silently
  aliased. MH prefix replay still requires exact request/observation equality.
- LFM's 65,536-wide output head contains IDs absent from its 64,402-entry
  tokenizer. Projection generation masks these padded IDs, and both target and
  guided likelihoods use the identical restricted/renormalized support. The
  scoring identity is now `closed-assistant-tokenizer-support-temperature-logprob/2`.
  This is a conditional distribution over decodable IDs, not the old unrestricted
  output-head distribution. Old receipts are never rescored or mixed in-chain.
- Canonical initialization excludes the native template's formatting newline
  after assistant EOS: actual generation stops at EOS. EOS itself remains scored.

V3 and v4 rounds were safely interrupted with receipts preserved after these
collector defects appeared. Their stopped-review receipts explain the changes;
no training publication. V5 is running on the local GPU:
`natlang-student-projection-mh-v5-diverse-20261004.service`,
`runs/student-posttraining-20261004/projection-v5-native-terminator`, plan SHA
`5079d52f240e015381ca3d3225cbfce09981ab323192fcb22cfa10100a95ca6e`.
It uses the immutable completed full-SFT adapter, eight suffix proposals per
case, fresh execution and native admission. First teacher replay admitted.
Seven targeted JS/Python checks passed, including asymmetric MH accounting,
strict prefix replay, padded-logit support, and canonical EOS formatting.

Next: review v5 yield/failures and family/label diversity; admit and register
verified candidates through the existing lineage/token/mix/source gates; prepare
a modest rehearsal-plus-projection phase with unchanged protected held-out rows;
then launch the resumable Muon phase via its reviewed manifest. No live
post-training phase has started. DGX self-improvement/Neuralese jobs remain
running and were not restarted; observed GPU utilization was 93%.

### 2026-10-04 projection failure review: agent guidance and replay identity

V5 reached 26 completed cases / 13 candidates / 87 native turns at the first
50-minute check; zero teacher replay failures. Render preview of v2's ten
candidate turns retained nine (616 supervised tokens); one repeated call was
explicitly unapproved and filtered. These are previews, not corpus publication.

Failed suffixes expose concrete ergonomics gaps: models read source handles
after moveTo, choose destination handles as the source, repeat identical failed
code, fabricate properties such as board.state, and sometimes convert blocked
plans into success after our rejection feedback suggests doing so. New folder
prompt guidance shows source-to-destination movement, destination verification
and unchanged handle paths. return_result feedback now repairs blocked/failed
reason/value fields and allows success only when the task is actually complete.
Existing frozen v5 remains unchanged; review these prompts in a later immutable
round, not by editing its runtime. Strict answer/file/honest-stop checks remain.

One live-inventory suffix had a prefix-observation mismatch. native/values.ts
uses a process-global liveId counter, also printed in agent livePreview. This
is a plausible fresh-replay nondeterminism source and remains to investigate
with an exact failing-request diff and task-scoped identity design. Do not
normalize away arbitrary observation differences or admit that rejected move.
Recovery training should emphasize API inspection, reuse of completed judgments,
checking state after failed writes, and honest stops; do not train raw failed
outputs as positive SFT or form DPO pairs without shared-context verification.

### 2026-10-04 live-identity replay defect confirmed and fixed

The train-only live-inventory case was replayed twice with identical teacher
actions and seeds. Both old runs passed the outcome oracle, but two of three
requests differed. With execution-local display identities, both outcomes pass
and all three requests are byte-identical. Evidence:
`runs/student-posttraining-20261004/live-identity-replay-review-v1.json`.

NativeRuntime now owns a display identity registry (shared by native invocations
within one public runtime task). Agent argument/scope previews, eval observations,
staged values and stored-local diagnostics use it recursively. Aliases retain
one ID within the execution. Process-global serialized trace identities remain
unchanged. Replay request equality remains strict; rejected historical moves
are preserved, not reclassified or migrated. Build passed, 83 targeted tests
passed, and an additional native-session fresh-replay regression passed.
Sealed runtime v42 (`runtime-v42-x64-student-recovery-r1`, manifest
`7bfff6cad6b9b161427a71e35c56481c2dc946d4a20dd9e45e79bba92c81affe`)
contains the fix and the preceding stop/file guidance. V5's frozen runtime is
untouched. Next collect a small targeted recovery round after v5 releases the
local GPU; compare execution yield without claiming broad model improvement.

Recovery round v6 is queued behind successful v5 completion and final-summary
publication; it holds no GPU resources while waiting. Unit:
`natlang-student-projection-mh-v6-recovery-r2-20261004.service`. Plan under
`runs/student-posttraining-20261004/projection-v6-recovery-guidance`, SHA
`623d932b771ff95c6889e60a4ea91750a2bc37506477d46bfa453976056dea90`.
Eight exact train-only references: two route-planning, two folder-criteria, two
live-inventory, one event-retry, one contract-diagnosis. Fixed student adapter
and search controls match v5; prompts/runtime differ deliberately. Source closure
is a pinned exact subset of the reviewed broad packet; preflight passed eight
cases with zero provider calls. Collector code is copied and pinned within this
round to keep later repository edits from invalidating its queued launch.
First planning attempt used a wrong family label and produced no plan/provider
calls; its failed transient unit is retained, r2 is the actual queued launcher.
V5 reached 42/128 complete, 21 candidate episodes / 156 turns; candidate counts
remain separate from published training data. 84 targeted tests passed in total.
