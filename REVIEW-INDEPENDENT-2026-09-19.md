# Independent review of natlang

Written on 2026-09-19, before opening HANDOVER.md. This assessment is deliberately frozen before reading that file.

## Judgment

This is a worthwhile research project with a functioning experimental system. The strongest idea is to make a small model operate on short, typed, explicit pieces of state, delegate exact operations to code, and compose semantic judgments through reusable functions. The implementation already makes that idea testable.

The central claim remains unproven: a small model can reliably interpret previously unseen pseudocode over many dependent steps, with enough benefit to justify having the model choose those steps. Tool learning is demonstrated; general interpretation and useful whole-program reliability are not yet demonstrated by the evidence I inspected.

I would continue, but make the next milestone a controlled experiment. Freeze language expansion, select one useful workload, establish clean evaluation, then train against the observed failures. More features and more synthetic data are currently easier to produce than decisive evidence.

## Scope and evidence

I inspected the README, specification, relevant planning/training material, runtime, tool surface, native decoder, reference policy, generators, training/evaluation scripts, tests, example programs, and saved evaluation logs. I did not run a new model evaluation or training job, change running services, or consult external benchmarks. Saved model results describe their recorded versions, not necessarily today's code or the unfinished v6 run.

Local verification:

- `pytest -q`: **97 passed in 26.40 seconds**.
- `python3 tools/check_conformance.py`: **29 files and 129 type expressions passed**.
- Small direct probes confirmed the implementation defects described below.

The tests provide meaningful evidence for harness mechanics and reference trajectories. Most substitute scripted policies or oracles for the model; they do not establish model interpretation accuracy.

## What is strong

1. **The state and function boundaries are coherent.** Typed inputs, private locals, lexical code bases, explicit return values, and short episodes reduce the amount the model must remember. Resumable partial results and provenance are useful research infrastructure.
2. **Exact work has an explicit home.** Crisp functions and map/fold/iterate semantics avoid asking the model to count, accumulate, or reproduce large values unnecessarily.
3. **The data pipeline executes its own references.** Checking generated trajectories against the real runtime, their grammars, and expected outcomes is much stronger than accepting plausible teacher transcripts. Rendering training examples with the inference template is also a sound choice.
4. **The project can complete a local training loop.** Export, fine-tuning, serving, and evaluation already work. That makes relatively small experiments practical.
5. **The hypothesis is falsifiable.** Program success versus length, unfamiliar structure, and wording can expose whether the architecture actually generalizes.

## Where the idea needs sharpening

The model has two distinct jobs: interpreting program structure and making semantic judgments at the leaves. A failure in either can produce the same wrong final answer. Evaluate them separately before choosing a remedy.

There is also a boundary to describe more precisely. The harness does not parse pseudocode or choose the next source statement, but it does execute map, fold, iterate, scheduling, and stopping semantics. “Owns no control flow” overstates the separation. The useful claim is that the model selects and binds operations while the runtime executes their fixed semantics.

Structural typing protects representation, not meaning. A complete `Report` can contain fabricated counts; a valid enum can select the wrong action. A line marked done is the model's assertion, not proof that the statement ran correctly. The saved tuned conformance log repeatedly demonstrates this distinction.

The most informative competing baseline is **the same semantic leaves under deterministic orchestration**, alongside one-shot inference with the same small model. If deterministic orchestration performs much better, the question becomes whether the authoring flexibility of pseudocode is worth the remaining reliability and cost gap. This comparison can preserve the project's thesis while testing its distinctive contribution.

For an initial practical demonstration, I would choose batch ticket triage with explicit rubric-based labels and exact aggregation. It exercises semantic work, composition, collection size, and measurable outputs. I would defer further webserver/highlighter expansion: they add substantial behavior without isolating the core advantage as cleanly.

## Evaluation is the first priority

### The loss holdout is not independent at the program level

`scripts/train_lora.py:79` shuffles individual exported turns and holds out the first 200. In the existing `data/sft-v5-small.jsonl`, reproducing the default split finds **195 of 200 held-out turns have another turn from the same program in training**. Later-turn histories can also contain earlier target actions.

This makes held-out loss an optimistic measure of generalization. It does not invalidate the separately generated seed-777 evaluation, which needs its own audit. Split complete programs before producing turns, and reserve program structures and wording families as stronger tests. Keep dataset, split, rendering, grammar, model, and code hashes with every run.

### The saved accuracy improvement is encouraging but not a paired comparison

The saved untuned/tuned turn logs report **29% versus 75% exact match**, on **168 versus 155 turns**, with different cell counts. Both score reply turns at 100%; those turns contribute to the aggregate. The evaluator samples only an initial part of the file and uses very small cells.

Use identical persisted sample IDs for every model, report non-reply actions separately, and distinguish canonical exact match from semantically acceptable actions. Teacher-forced next-turn accuracy is diagnostic; autonomous program success should decide progress.

### The recorded whole-program evidence is weaker than the summary

`runs/tuned-conformance.log` contains **5 yes, 13 no, and 3 unjudged** across 21 programs. `PLAN.md` says six correct. Reconcile the artifact/version mismatch rather than treating either figure as the current score.

Some outcome checks miss the behavior they intend to test. For example, program 14 passes grading when handed the scalar `2`, even if no records were emitted. The model evaluator passes only status/value/note to the grader. Grade emitted records and relevant state transitions as well as outputs. For general tasks, accept valid alternative execution paths; require particular behavior only where the test actually concerns that behavior.

### The next evaluation should locate the failure

Run four configurations on the same programs:

- Reference orchestration + reference leaves: harness/oracle control.
- Model orchestration + reference leaves: interpreter competence.
- Reference orchestration + model leaves: semantic competence.
- Model orchestration + model leaves: actual end-to-end behavior.

Add one-shot inference as a practical baseline. Stratify by program length, nesting, collection size, unseen compositions, renamed symbols, and paraphrased instructions. Include missing-information cases and adversarial text in inputs. Measure correct completion, incorrect completion, justified blockers, invalid calls, total tokens, retries, and wall time. Channel separation is useful but is not evidence of resistance to instructions embedded in data.

## Concrete implementation findings

**High priority — execution is not fully bounded.** `natlang/js.py:43` explicitly disables the time limit for effectful code. `natlang/runtime.py:453` exempts `mark_done` from the action budget; a probe issued 100 successful mark calls with zero charged actions. The agent loop has no independent overall turn limit. The specification's episode token budget is also not implemented in that loop. Add a hard turn/token/wall-clock budget covering all calls and retries, plus a killable worker for effectful JavaScript.

**High priority — a rejected operation can already have changed state.** At `natlang/runtime.py:465`, the write/call executes before the accompanying `done` range is fully validated. A probe writing `return=1` with `done=999` returned `rejected` while leaving `return=1`. `ToolAgent` may then silently resample the rejected call. Validate the complete mark first and make rejection semantics explicit; do not retry an effectful operation on the assumption that rejection means nothing happened.

**Medium priority — the advertised TypeScript contract exceeds implementation.** `spec/SPEC.md` promises type stripping and static checking. `js.run` directly executes JavaScript; `const x: number = 1; return x;` fails with a syntax error. PLAN acknowledges missing static checks. Either implement the promised subset or document the current executable language as JavaScript with typed frontmatter. Make proposed guarantees visibly distinct from implemented ones.

**Medium priority — dictionary helpers mishandle valid string keys.** `natlang/prelude.js` builds grouping/counting/indexing results with ordinary `{}` objects. A probe of `countBy(["constructor", "__proto__"])` returned a function-derived string and lost a key instead of numeric counts. Use own-key-safe storage and test reserved property names. These helpers sit inside the supposedly exact part of the system.

**Medium priority — performance and durability claims need implementation-level qualification.** Maps execute sequentially and copy values deeply. The tree and journals are in memory. The specification describes batching, lazy access, and stronger persistence/provenance behavior than the current core establishes. Report measured serial performance now; add batching after there is a stable correctness baseline.

The runtime is also accumulating several historical interfaces and semantics. After the measurement fixes, reduce this maintenance burden by identifying the supported surface and reconciling its documentation and tests. I would avoid a broad refactor before the next experiment.

## Recommended next work, in order

1. **Freeze an experiment contract.** Pick triage as the primary workload, pin source/data/model versions, define splits and acceptance criteria, and preserve the current model as a baseline.
2. **Make evaluation decisive.** Implement program-level splits, a fixed paired evaluation manifest, the orchestration/leaf diagnostic matrix, effect-aware grading, and machine-readable results. Turn the existing planned generalization tests into executable gates.
3. **Fix the boundedness and rejection defects.** Add narrow regressions for repeated marks, effectful timeouts, rejected writes/calls with marks, and dictionary keys. Align documentation with current guarantees.
4. **Run targeted training experiments.** Measure the current model first. Then rebalance only the deficient skills, add verified recovery trajectories from actual model failures, and compare with the frozen baseline. Inspect how many long examples are dropped by the 3,072-token cap; otherwise training may systematically omit the hardest contexts.
5. **Require a whole-program gain before expanding.** The training plan's proposed 25-call/70% program-success milestone is a reasonable research gate to operationalize on genuinely held-out programs, with adequate sample size and explicit blocker handling. It is not a production reliability target. Report the entire success-versus-length curve and compare cost/quality with deterministic orchestration and one-shot inference.
6. **Optimize the demonstrated bottleneck.** If orchestration works and latency dominates, batch independent leaves. If leaf judgments dominate errors, invest in domain data and evaluation. If novel structure remains weak, simplify the supported dialect or offer an optional compiled subset while preserving a clear test of model interpretation.

My recommendation is to keep the architecture and tighten the experiment. The next persuasive result is a reproducible demonstration that the small model correctly executes unfamiliar useful programs, with measured costs and an explanation of where failures originate.
