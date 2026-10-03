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
