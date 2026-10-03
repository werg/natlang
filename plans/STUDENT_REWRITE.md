# Student rewriting after direct SFT

## Decision — 2026-10-03

Use a cheap student-conditioned rewrite lane first. Keep ordinary completion-only
SFT and Muon. Do not implement MCMC or likelihood ranking in the first experiment.
User explicitly prefers quick/simple improvements over straightforward SFT.

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
Our first lane is that cheaper baseline, with stronger execution admission.
There is no claimed reproduction of the full sampling algorithm or its guarantees.
Their reported models are3B/7B, not our350M; applicability is an experiment.

## Existing integration

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
2. Launch a small rewrite cohort against that snapshot. Store admitted candidates
   as useful training data; do not throw away this collection as a pilot. If few
   valid variants emerge, shorten tasks/reference hints or focus on leaf calls.
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

## If cheap rewrites help

Next add ordinary-prompt likelihood scoring and choose the better verified variant
among one or two candidates. Score only student completion tokens, under real
contexts with no teacher hints. Current model-turn transport has no logprobs;
implement a direct pinned HF scorer rather than use privileged generation scores.
Keep this separate from exact MH: average-token greedy selection is a heuristic,
not the sum-log-probability/proposal ratio required by the theoretical sampler.

Multi-turn rewriting stays at runtime action boundaries. Token-block splicing
through JSON/code/tool calls or editing actions while preserving old observations
would fabricate trajectories and is forbidden.
