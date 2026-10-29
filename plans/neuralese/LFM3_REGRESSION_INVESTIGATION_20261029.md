# LFM3 capability regression: investigation brief for independent review

## Operational update, 10:05 Berlin

The original brief below is a timestamped account, not current launch state.
The overall task regression is still unexplained; none of these engineering
findings establishes restored capabilities.

- The convolution fix and its BF16-only, sparse/empty-invocation and fast-path
  follow-ups are integrated in main through `cc3551dbd`. The full-size cached
  comparison and an actual8341 FP8 shared-graph update-schedule comparison passed;
  the latter is not graph-versus-cached parameter-gradient proof. Integrated
  shared-state/optimizer/focus checks passed95 tests.
- A further objective bug was confirmed: balancing by file SHA changes weights
  when identical data is divided into different numbers of files. The old
  declared cap8 became actual maxima around15.2 after normalization. Current
  20.92% communication rows received only1.748% of one-visit graph-weight mass.
  These are graph-weight figures, not token or realized-gradient percentages.
- Source-bound logical maps and bounded mean-one normalization are integrated
  through `2b12a1579`;28 focused integrated objective/map/focus tests passed.
  The first corrected policy deliberately uses only authenticated mechanism
  classes: communication, two identified coding sources, and remaining ordinary
  sources. It does not pretend to be a curated semantic taxonomy. Actual cap8,
  mean1; further task-family refinement remains work.
- Peer1 resumed from its earned13846 checkpoint with corrected convolution,
  weighting,38 prompt variants,57 focused examples and fresh EMA. Saved13887
  proves all57 received one new visit (including all38 variants), unchanged old
  exposure credits, new objective `cf3fd663...` and41 EMA updates. This is actual
  saved training, not capability lift. Other peers are being changed one at a time.
- Exact evidence: `prompt-variants-pilot-v1/source-weighting-audit.json`,
  `rechunk-weight-sensitivity.json`, `mechanism-family-proposal-v1/report.json`,
  `peer1-13887-saved-state-audit.json` and `peer1-13887-focus-exposure.json` under
  the data directory named below. Full/control exports and independent owner GET
  remain distinct obligations; consult actual receipts for current status.
- The newer local full-state12957 recovery is qualified under
  `/mnt/external/natlang-training-recovery/lfm3-v3-live-20261028/peer1-step12957/`.
  Its GGUF export is running for refreshed evaluation. A first14GiB export job
  hit its own memory limit; the bounded28GiB retry is separate. No new capability
  score has been earned yet.
- A complete predecessor-source census found131,666,941 own-output target tokens;
  the two identified coding sources supply17,701,133 (13.44% unweighted one-pass
  target supply). This is not realized weighted SGD exposure or the entire coding
  provenance census.
- Actual first1024-batch sequencing has2398/4313 adjacent row pairs with the same
  authenticated named task template, including1813/3290 within-batch pairs.
  Missing identifiers remain unknown. The first bounded batch-reordering
  prototype gave little benefit and is not deployed; diversity work continues.
- Owner explicitly requires early JOLT+ with full explanatory prompts, complete
  schemas and valid idiom examples. Executable on-policy preparation, current
  protection/oracle checks and sandbox Node/collector staging are underway.
  No actual JOLT+ policy update is claimed. The eight requalified training cases
  are eight cases, not the full planned curriculum.
- The old b37 full-check attempt was preempted and restarted; it did not complete.
  Its replacement tests frozen `cc3551dbd` plus a recorded WIP overlay, ledger
  `natlang-main-frozen-cc355-1029`, log `check-cc3551dbd-full.log` under the ops
  directory. It does not include the later logical-weighting module changes.

## Original investigation snapshot

Status captured 2026-10-29, approximately 09:10 Europe/Berlin. This is a
collaborative investigation brief, not a claim that the regression is understood
or fixed. Supervision and training continue. Read the actual files and live
state before acting; several investigations and transfers are still running.

## What needs explaining

On eight matched graph-search development worlds, the untrained instruction
model passed **6/8**, while our trained **step8341 passed 1/8**. This is a serious
measured regression on this task family. It is not established that every
capability, every model line or the newest checkpoints regressed to that extent.
There is no satisfactory causal explanation yet.

The owner explicitly suspects a larger problem than prompt sensitivity or the
convolution bug described below. Treat those as findings within a broader
investigation. Do not assume that fixing either restores the model, or that
insufficient idiom examples explain deterioration below the base.

We want the best useful models promptly, not clean experimental runs for their
own sake. Changes, interruptions, continuations and justified redos are allowed.
Preserve useful full states and protected evaluation worlds. Base-relative KL
is telemetry, not a capability criterion or a reason to prevent improvement.

## Initial baseline and intended recipe

Model: `LiquidAI/LFM3-1.7B`, pinned snapshot
`40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.

Baseline report: `plans/neuralese/LFM3_1P7B_BASE_EVAL_20261027.md`.
That older, different held-v3 packet produced 13/16 successes, or 13/15
scorable tasks: two semantic failures and one quarantined task with zero model
requests. All269 model replies parsed as tool calls, with no runaway repetition
or budget exhaustion. This is useful historical context, **not** a matched
broader evaluation of the current trained checkpoints.

Baseline weaknesses included doing per-item judgement and sorting in reasoning;
zero uses of `nl`, `iterateOn`, `decide` or `.sort()` in 204 evals; input/service
binding collisions in 10/15 tasks; guessed function names; unawaited NL calls;
and long child reasoning. We intended to improve those habits while teaching
the entire native Neuralese architecture.

Read `LFM3_RECIPE_20261028.md`, `LFM3_COMMITTED_DECISIONS_20261028.md`,
`LFM3_PORT_20261028.md` and the actual current trainer/options. Some design
paragraphs describe proposed stages rather than the exact current run.
Current four learners are in joint SFT; they are not earning RL updates or
performing distributed outer merges at present. Later stages include whole-graph
RL/JOLT+, program tuning and the LoRA-bank metalearner. The newer LFM3 recipe
does not wait on Mellum or cross-model VAE work.

## Exact failing evaluation and behavior

Protected development packet:
`runs/natlang-dev-task-packet-v2-20261028/cases.jsonl`, SHA256
`6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560`.
Matched worlds are original cases80–87, all `curriculum_iterate_frontier`.
Spec: `/home/werg/data/dgxops/claude-recovery-20261028/data/dev4751-paired-base-80-87.json`.
Budgets:1000 model requests,1000 turns,7200 seconds/case, eight workers.

Trained evaluation directory:
`/mnt/external/natlang-training-recovery/lfm3-v3-live-20261028/peer1-clean-current/paired-dev-80-87-keep-v1/`.
Read `probe.log`, `probe/task-eval/`, server log, template kwargs and the
individual `jobs/*.result.json` trajectories, not only the summary.

| Original case | Step8341 result | Requests |
|---|---|---:|
|80|semantic mismatch|3|
|81|semantic mismatch|997|
|82|success|997|
|83|semantic mismatch|3|
|84|semantic mismatch|997|
|85|semantic mismatch|997|
|86|semantic mismatch|3|
|87|semantic mismatch|3|

Wall4893.5 seconds; seven semantic mismatches. The base passed 6/8 on the matched
worlds; its six successes took at most45 requests. Recheck original base run
artifacts before broadening the comparability claim to every sampling or serving
detail. An older step4751 run stopped after 29 completed tasks, seven accepted;
that partial run is not a matched broad regression estimate.

In graph80, the trained model initially emitted a huge unfinished XML `eval`
call containing a repetitive regex beginning approximately:
`graph.search(/n-gleng|n-mait|links|path|route|route|path|route|link|node|graph|tree|map|list|array|object|function|boolean|number|string|...)`.
It consumed8192 output tokens without closing the call. After cutoff feedback,
it queried only the start node's neighbours and prematurely returned `null`;
the oracle expected distance3. A short retained reasoning field does not mean
the generation was short: the repeated material was unfinished tool code.

## Model and serving identities

Qualified full checkpoint8341:
`/mnt/external/natlang-training-recovery/lfm3-v3-live-20261028/peer1-clean-current/checkpoint.pt`.
SHA256 `1b74798966c659b19c305bb84c7aa4d9c6576afae9f8b16d6ff7696b8c1f2c09`,
6,584,311,987 bytes, with optimizer/RNG/exposure. Older2512 and 4751 remain.

Its BF16 GGUF is `dev-gguf-bf16-v1/model-bf16.gguf` below the same directory:
SHA256 `616f28670a0c857d75a19fee46c267bb8c142bbf292d0237010fcf783a24dceb`.
Qualified actor binary:
`/home/werg/data/dgxops/claude-recovery-20261028/data/llama-actor-worktree/build-cuda-121/bin/llama-server`,
SHA256 `cb57d2b2769769cd080c76321988b1c4cd34e4217a4146b3b8cb51fd5fb591b9`.
Other llama builds are not automatically interchangeable evidence.

Native/GGUF comparison on the actual repeated-code context:32 sampled positions
initially agreed; a longer256-position comparison agreed on254 top predictions,
including every192 tail position from64 onward. Some low-probability logit/logprob
differences remain. This supports repetition being present in native computation
as well, rather than an obvious serving-only cache failure. It is not proof of
whole-task/backend equivalence.

Receipt:
`/home/werg/data/dgxops/claude-recovery-20261028/data/step8341-repeating-code-parity-20261029.result.json`.
Raw first response: `step8341-greedy-response-20261029.raw.json` in that directory.

## Prompt/template findings: a real mismatch, not a complete explanation

Main audit directory used below:
`/home/werg/data/dgxops/claude-recovery-20261028/data/training-bug-audit-v1/`.

The tokenizer's chat-template file is byte-identical to the pinned official
snapshot, SHA256 `f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818`.
Its default is `preserve_thinking=false`; our training policy intentionally
retains previous thinking. An earlier serving launcher omitted the manifest's
thinking kwargs. That was fixed before the8341 matched evaluation, which still
regressed. File identity alone does not certify all prompt construction.

Nineteen actual verified ordinary training actions reproduce under our historical
**tool-less program-header** rendering, rather than full rendering. Example:
2762 total tokens historically;3688 for a compact variant retaining five schemas;
6816 full. All carry the same370-token assistant target in that example.
The audit's direct `apply_chat_template` comparisons also contain differences
from additional project prompt shaping/emulation; inspect those fields rather
than treating file equality as complete rendering parity.

For failing graph80's exact initial context:

| Rendering | Prompt tokens |
|---|---:|
|Full with five schemas|4533|
|Program-header retaining schemas|1326|
|Historical training-style header without schemas|400|

Native full token IDs exactly match the captured full HF/llama prompt in that
comparison. No new Neuralese marker IDs125033–125036 occur in it; no accidental
marker injection was observed. `eval/task_cadence.py` defaults to full rendering,
while historical training/harness header mode removes schemas.

Controlled same8341/binary/context greedy512-token first-response experiment:
full prompt repeated an incomplete call; header with schemas produced a complete
61-token call; header without schemas a complete52-token call. This establishes
prompt sensitivity. It does not establish task success.

A second real bug blocked compact execution: with no API tools, llama returned
complete XML in assistant content rather than structured `tool_calls`; the TS
transport rejected it. After the parser repair, graph80 executed about 61 requests
and still returned `null` instead of 3. Four further worlds shared one server slot
and each hit600-second wall budgets after 6–7 requests. Those four results are
inconclusive because queue time and long generation were combined.

Relevant files: `eval-prompt-contract.json`, `ordinary-render.json`,
`paired-prompt-results.json`, `paired-prompt-parser.json`,
`held-world-program-header-fixed-v1/`, `held-world-program-header-four-fixed-v1/`.

## Parser/conversion repair status

Integrated in main via merge `cc3971b6e`, helper tip `5d65afe01`:
complete XML calls authenticated against response schemas retained outside the
model prompt. Actual provider `raw_calls` remains empty; raw request/response,
canonical hashes, parser-policy SHA and parsed action custody are retained.
SFT materialization, offline-pair production/DPO, failure-repair records, whole-
graph RL replay and tuning conversion now propagate and revalidate that custody.

Entry points: `ts-host/src/model/native-xml-{calls,custody,policy}.ts`,
`ts-host/src/model/chat-completion.ts`, `ts-host/src/runtime/trajectory.ts`,
`training/neuralese/natlang_neuralese/data/native_xml_custody.py` and
`tests/neuralese/test_native_xml_custody.py`. Parser implementation SHA is
`1b2713adb032003d3522794fe65b75f03d9b31a1453bc86027f544f460bc315e`.
Local JSON-schema refs, boolean schemas, structural object equality, nullable
unions and typed additional properties are covered; external refs/unknown
keywords fail closed. Independent review of schema coverage remains welcome.

Integrated checks:41 Python tests passed across custody, failure-repair DPO,
pending focus and coordinator mapping;63 TS parser/materializer tests passed.
The first direct TS invocation used the wrong dist working directory and failed
module resolution; rerun with explicit `NATLANG_TEST_DIST` passed. Captured graph80
XML is covered using explicitly synthetic transport envelopes in some tests:
**not** a newly executed oracle-approved repair pair or a capability improvement.
No production training runtime switch has been made from this merge alone.

## Training/data audit and remaining concerns

Step8341 source:478664 rows, SHA256
`3633c0fee79abe57f865715b0c995858379aabe379ab16a39e74611a76f365ad`.
Current source:591119 rows, SHA256
`1aa7a01ad7a1b589d575a4ef30ea33d2069a5af27ec8f86dcd07b668149ae7e3`,
adding112455 Big2 communication graphs. **8341 had no exposure to that append.**

Historical source corrections removed2081 held rows and nine quality rows from
the earlier480754-row source. Historical exposure is retained honestly: examples
include11 held visits on peer1 and 381 on peer4, so those historical held sets are
diagnostic rather than pristine. Do not erase exposure history or mix protected
related worlds into a repair dataset.

Audit folder `graph-search-audit-8341/` beside the main audit contains exact
source hashes, checkpoint bindings, exposure and motif inspections:

- All513 current leaves were hash-checked, covering591119 rows. Exact
  `graph.search` variants and the exact repeated enumeration were not found.
  That does not exclude more subtle harmful patterns or incomplete audit coverage.
-8341 had33414 distinct visited rows and 445250 unvisited of 478664, once each
  in the audited counts. Registered data is not exposure.
- Of1680 active bound `iterateOn`-related rows in the extract,297 were visited,
  1383 not. Sixty-seven frontier candidate rows had zero recorded visits in 8341
  and in the later audited post-adoption checkpoints. Only19 of those were
  independently exact-oracle/actor qualified for the immediate focus request.
- Token-mass cohort fairness was repaired and is live, but candidate frontier
  rows could still appear thousands/tens of thousands of packed groups later.
  A one-time focus mechanism now preserves complete packed groups and quotas
  while bringing qualified rows forward. It is not yet deployed.
- Broader regex/motif inspection found exposed benign examples and unvisited
  redundant/poor examples; there was no basis to attribute the exact failure to
  an observed memorized bad target or to blanket-quarantine the whole corpus.
- Actual ordinary masks on19 examples have zero context/history supervision,
  reply mass0.9 and reasoning mass0.1. Five real Big2 edges/ten node sequences
  reproduced source tokens and masks. These are samples, not a complete quality
  proof for every training row.

Historical converter risk: older projected/static-pair conversions sometimes
replaced a complete typed producer response with only its exported payload,
teaching incomplete returns. That family was quarantined/repaired earlier; audit
the actual current lineage and remaining descendants rather than assume prior
repair proves every source is correct. See AGENTS.md's complete-producer-target
and source-correction sections. Old damaged exposure can persist in weights.

New robustness data:38 action variants,19 full/19 compact, preserve exact original
assistant targets and actor/source/oracle custody. All40 original pilot variants
(including two chain variants) passed protection-union and exact-content dedup
against591119 train and 5912 protected rows. The chain variants are separate
because of numerical qualification. First action successor591157 rows:
SHA256 `9e075c6b4ccfc53fc8122138acbce34d041c13dfa6bcab189585504aeaad4a1d`.
Combined focus is57 rows:19 originals plus38 variants, **not67 originals**.
Ten further action variants across five task families have also passed protection-union, CE and dedup checks;
additional authentic chain families are being prepared. Neither is exposed.

Data state:
`/home/werg/data/dgxops/claude-recovery-20261028/data/prompt-variants-pilot-v1/`.
Read `protection-union.json`, `content-dedup.json`, `action-successor-v1/` and
`four-action-prepared-v3/receipt.json`. Plans retain full/foundation role coupling,
actual predecessors and peer2's retirement proof. A real source+CE+focus handoff
composition gap was fixed in main `e14dd0476`; test commit `f941827fa` remains on
the helper branch pending integration. No source/CE/focus activation yet.

## Optimizer, precision and model-correctness checks

Actual settings audited: decoder backbone lr2e-5; learned sparse/interface rate
1e-3; rank8; MuonSR hidden matrices and stochastic-rounding Adam-family ordinary
parameters. Parameters and optimizer histories are BF16 under the current owner-
selected profile, unlike older FP32 sparse-row policy. Embedding/tied readout and
final norm follow the frozen backbone policy; decoder layers and native interface
train. Consult actual checkpoint options and `train/backbone_policy.py`.

QAT is substantial: BF16 weight1, FP8 weight0.5, NVFP4 weight1. Fully ramped
sampling probabilities are40%/20%/40%; sampled loss is multiplied by total2.5 as
the declared weighted-sum estimator. NVFP4 covers MLP weights/activations with
saved static activation calibration; FP8 covers its declared attention/MLP points.
This is not proof that the mixture, ramp, calibration, or rates preserve behavior.
Do not describe optimizer scale invariance as a literal2.5x learning-rate proof.

Completed checks and their limits:

| Inquiry | Finding | Limit |
|---|---|---|
|Pinned-base FP32 port vs official implementation|701 tokens; max logit difference about 4.39e-5, same top1|Sample, base weights|
|BF16-vs-FP32 baseline numerical scale|Mean KL about 0.000657 vs reference0.000661|Not a trained-model capability check|
|FP32 gradients vs official reference|Ten conv/attention/norm parameters,321 tokens, worst relative max about 2.64e-6|Not full BF16 shared-graph backward|
|Saved8341 reconstruction|291 model/interface tensors,1,990,990,592 elements exactly equal after reload|Not live-before-save identity|
|Real8341 update schedules, BF16|Same loss, all 153 Muon gradients and every trained parameter exact|One2762-token ordinary row, controlled SR draws|
|Same update-schedule check, NVFP4|Same exact result using saved calibration and 2.5 loss scaling|Same row; no FP8 check; no whole-graph proof|

Real-update scripts/results: `training-bug-audit-v1/real-update-parity.py`,
`real-update-parity.json`, `real-update-parity-nvfp4.json`. The latter records the
NVFP4 precision explicitly; a copied prose scope string still says "not QAT" and
is stale—inspect executed script, precision field and journal rather than that
phrase. The diagnostic fixes seeds and parameter-addressed SR draws; it preserves
saved optimizer histories but does not reproduce the live learner's next RNG draw.
Ledger units: `natlang-real-update-parity-1029`, `natlang-real-update-qat-parity-1029`.

**Unfinished:** actual same-boundary live-before-save versus reload comparison;
broader trained/base task comparison; raw versus EMA choice and annealed release
behavior; FP8 correctness/quality; LR/weight-decay/row-rate/QAT damage; complete
target/exposure/quality census and true full-recipe coverage. These are not cleared
by matching a base model, saved tensors, scalar loss or gradient norms.

A separate routing concern identified by inspection: `make_native_optimizer`
passes native interface matrices into the generic shape-based `latent_split`.
Current rank8 adapters are narrow and remain AdamW, but rank>=16 could be routed
to MuonSR contrary to the native ordinary-parameter rule. This is not a cause
established for these rank8 runs and has not been repaired here. Existing explicit
ordinary-Muon→AdamW migration currently handles the generic Muon child, so do not
silently reinterpret an affected MuonSR history when fixing the broader issue.

## Newly isolated shared-state backward bug and candidate fix

This is a confirmed mismatch and a locally demonstrated fix, **not a proven
cause of the observed6/8→1/8 task regression**.

Real source-bound e799 chain:10 nodes,12 edges/imports, five producer spans.
Compact variant36771 physical tokens, SHA256
`1330f1ad4fe3cdfe68703d60760fa204ffcdf27460bf97f9136a9d7e9f8c19cc`.
Full variant68767 tokens has not completed numerical qualification.

Frozen native runtime1f0 produced bit-identical measured consumer outputs versus
explicit cached execution but different gradients with respect to the36771x2048
input embeddings: global relative L2 about 0.00663353, cosine0.999978,
max absolute0.0008544921875. Producer n0 differed (relative L2 about 0.013636);
the other four producer spans matched exactly. A prior25609-token fixture had
similar localization/differences; a similar prior result is not a correctness proof.

Initial hypothesis was n0's fan-out to eight consumers. Controlled n1-only loss
still differed, while n2-only matched; simple FP32/BF16 summing of independently
rounded gradients did not explain the combined VJP. Layer localization then
showed exact gradients through the first attention layer (three layers total)
and first divergence at the next convolution layer (four layers total).

Cause found: graph convolution gathered all invocation histories together;
its BF16 scatter/backward accumulated repeated histories across flattened
invocation boundaries, changing rounding relative to per-invocation cached
execution. Candidate retains invocation boundaries in the gather while keeping
one layer-major graph forward; it does not replay the graph or relax tolerances.

Candidate commit `9c80b606c` in
`/home/werg/data/dgxops/wt-bf16-graph-gradient-1029`, based on `cc3971b6e`.
Changes `model/shared_state.py:_convolve` and adds a focused regression.
Full real ten-node rerun: **zero** measured output difference and **zero**
difference across all 75,307,008 input-gradient elements; all five producer spans
receive nonzero gradients. Weights/RNG unchanged, no SGD/exposure credit.
Focused convolution test and six shared-state/checkpoint CPU/CUDA/model-family
checks passed. Peak CUDA allocation58,379,756,032 bytes (~54.37 GiB).

Evidence directory:
`/home/werg/data/dgxops/claude-recovery-20261028/data/bf16-gradient-investigation-1029/`:
`fixed-diagnostics.json`, `fixed-gradient-arrays.pt`, `qualify-full.py`,
`layer-localization*.json`, and localization scripts.
Original arrays/receipts: `prompt-variants-pilot-v1/header-numerical-strict.json`,
`header-gradient-arrays.pt`, `header-fanin-diagnostic.json`, `header-ulp.json`.
Ledger: `natlang-bf16-graph-full-fixed-1029`, `natlang-bf16-convolution-unit-1029`,
`natlang-bf16-graph-regressions-1029`.

Important scope: this full-size test uses the pinned base in BF16 with frozen
weights and an objective on consumer final hidden states, differentiating input
embeddings. It is **not** a full actual8341 model-parameter/QAT/optimizer update
comparison or a post-fix task evaluation. Review full packed graphs, subset/actor
paths, empty invocations, state-stop gradients, prefix sharing and dtype effects.
The candidate branch is not merged or deployed at this document's timestamp.

## Current operations and where another agent can help

Four live LFM3 learners, most recent inspected steps13414/12364/12781/11440;
all advancing, no observed cgroup OOM kills. This is volatile state, not a promise
that those are current when read. Authoritative live health:
`/home/werg/data/lfm3-sft-20261028/production-v3-live/standalone-live-health.json`.

|Role|Sandbox|GPU|
|---|---|---|
|peer1|`f3ad785f-9f0b-4716-905a-ed26f0a83e1e`|H10080 GB|
|peer2|`ea17666b-084f-4236-b552-48efd43f9f74`|RTX PRO6000~96 GB|
|peer3|`3aba3399-4a8d-4e48-b8a9-069e653bf0ef`|RTX PRO6000~96 GB|
|peer4|`b1662ef1-2514-49f4-be56-645e83a6712c`|RTX PRO6000~96 GB|

Each has16 GiB host memory/one CPU. Four standalone runs currently have separate
optimizer/RNG/exposure histories and no data shard/outer merge. Restored DiLoCo
coordination is separate work: old outer state was lost in a same-ID workspace
reset. Do not invent old outer momentum, silently average incompatible states,
or restart learners from an old parent. Full-layout recovery qualification is
being improved under strict memory guardians; small64MiB success did not qualify
full-model operation. This may affect utilization/duplication but is not itself
evidence explaining the local8341 behavior.

Checkpoint exports use independent private PUT/complete GET/hash and workstation
readback in addition to snapshots. Quota can cut off snapshot access without a
sandbox. Newer recovery/evaluation local copy12957 is being retrieved, but at the
latest report an older10799 GET was stalled/slow and delayed the serialized12957
copy. Neither a requested GET nor a staged file is a verified local checkpoint.
Peer3/12791 and peer4/11417 just gained cloud-side full GET verification; owner
readbacks remain separate. Preserve qualified8341 and earlier required parents.

Current engineering ownership:

- Primary Codex: supervision, integration, independent review, production decisions.
- `terra_recover_claude_work`: implemented XML custody and isolated convolution
  candidate; latest bounded work complete, available for coordination.
- `terra_lfm3_data_recovery`: prompt variants, full held-union/dedup, actual source/
  CE/focus continuations, broader verified-family expansion; source work ongoing.
- `terra_qat_recovery`: healthy learners, exports/recovery, staged action/focus
  rollout and bounded coordinator full-layout work. No rollout authorized from
  a merely staged package or old-parent capacity receipt.

Most useful independent Claude investigation: challenge the **whole causal
story**, not only this convolution fix. Reconstruct the actual8341 lineage,
target construction, trained parameter changes, precision/QAT, sample weighting,
exposure, prompt layouts and evaluator identities. Audit whether the existing
checks compare both sides through the same faulty code. Compare capability on
broader matched tasks and fresh qualified weights, protecting held data. Review
the convolution fix's model-parameter gradients and complete supported execution
paths. Propose/implement the changes most likely to restore and improve actual
behavior; neither additional data nor a numerical fix counts as restored ability
until task outcomes show it.

## Independent Claude findings (2026-10-29, ~10:00): the weights are random-walking

Workspace and scripts: `/home/werg/data/dgxops/claude-regression-1029/`
(`window_cos.py`, `sr_noise.py`, `mass_census.py` + `mass-census.json`, `interp.py`,
`run_eval.sh`, `eval-*/`). All measurements use the actual checkpoints and the
trainer's own `own_output_supervision.graph_weights`.

**1. Successive training windows move the weights in unrelated directions.**
Over 142 hidden matrices of the peer1 lineage, the cosine between the weight
deltas of disjoint windows is mean 0.029 (0→4751 vs 4751→8341), 0.002
(4751→8341 vs 8341→12957) and 0.016 (0→4751 vs 8341→12957). Total displacement
0→12957 is 8.84 (MLP matrices, Frobenius) against window displacements
5.04/4.60/5.41: the root sum of squares is 8.71, the coherent sum would be 15.
The trajectory is diffusion: under 3% of each window's movement is a direction
the next window repeats. The no-signal part keeps growing like sqrt(steps) (rms
of the exponent-independent delta 1.5e-3 at 4751, 2.05e-3 at 8341).

**2. About 35% of the weight-change energy is BF16 stochastic-rounding noise.**
With lr 2e-5 the Muon per-element update is ~0.2·lr = 4e-6, about 1/30 of the
BF16 spacing of a typical weight (~1.2e-4 at |w|≈0.02). Stochastic rounding is
unbiased but adds variance u·s per step, so its accumulated energy is
proportional to each weight's BF16 spacing s. Bucketing elements by base
exponent: E[d²] is flat for small weights and rises linearly with s for large
ones (fit slope 0.0126 at 4751, 0.0224 at 8341; the SR model predicts ≈N·|u|,
0.019/0.033). Six matrices, consistent noise share 0.31–0.39 at both steps.

**3. The remaining movement is small-batch noise amplified by normalized
updates.** Exposure counts: 33,414 graph visits over 8,341 updates = **4.0 graphs
per optimizer update** (telemetry: 24–32k forward tokens, often only 340–3,400
supervised tokens per update). MuonSR (and AdamWSR) apply a fixed-size step
whatever the gradient's signal-to-noise, `step_in_backward` makes every
32k-token micro-batch one update, and there is no gradient accumulation. With
momentum 0.95 the measured window cosines imply roughly 1% signal per step.
Every peer is doing this independently; with no outer merges, nothing averages
the noise away.

**4. Behaviour tracks the noise, not a learned skill.** Same capped spec
(150 requests/turns, 3600 s, cb57 server, keep-thinking kwargs, 8 slots):

| Weights | Successes | Behaviour |
|---|---:|---|
| base | 6/8 | `read_code(graph)` then BFS-style exploration, ≤45 requests |
| base + 0.25·(8341−base) | 5/8 | direct `graph.links` code, short turns |
| base + 0.5·(8341−base) | 2/8 | every case: one request, `return_result(null)` |
| 8341 | 1/8 | 4 cases: 8192-token runaway call then null; 4 cases: the same ~45-token `graph.search` call repeated 147 times |

Shrinking the delta 4× (noise energy 16×) is as good as base; the full delta
produces repetition loops, the classic damaged-weights signature.

**5. The objective concentrates mass on one-step answers.** Full census of the
478,664-row 8341 source under the actual objective and family scales:
85.8% of loss mass is on invocations with a single decision; trajectories with
10+ decisions hold 14.2% of decisions but 0.9% of mass. `return_result` gets
39.4% of all mass. Replies ≥512 tokens are 21.9% of supervised reply tokens but
0.9% of mass (<32-token replies: 10.4% of tokens, 40.9% of mass). Source-cohort
family scales span 0.17–14.9 although `max_scale` is 8: `family_scales` caps
before renormalizing, so the cap does not hold. This shapes *what* the small
signal teaches (answer at once, little long-code practice) and plausibly the
α=0.5 immediate-return mode. It is secondary to the noise.

**Why the existing checks did not catch it.** The real-update parity checks
compare the trainer with itself and confirm that each update is computed as
designed; they cannot see that the designed update is 4-graph noise. Matrix
norms stay flat because a random walk mostly rotates. Held CE is a single
number per eval. Lower train loss on 4-graph batches is expected either way.

**Ruled out here:** tied `lm_head` (identical storage to `embed_tokens` in
8341); target alignment (query `target-1` predicts token `target`); the
convolution rounding order (≈1e-3 relative accumulation-order differences,
orders of magnitude below items 1–3; fine to merge as oracle fidelity).

**Changes (owner 2026-10-29: "let's make the change"; Claude is implementing
them, and restart decisions belong to the agents):**

1. Accumulate gradients to ≥64 graphs per update (≈16 micro-batches of 32k
   tokens) at the same lr. SR energy per epoch scales with the number of
   updates, so this cuts both noise sources ~16× while keeping the signal per
   sample. The gated `step_in_backward` already lets backwards outside
   `in_backward()` accumulate `.grad`; the last micro-batch steps.
2. If SR noise still shows in the exponent-bucket fit afterwards, add Kahan
   compensation (a BF16 error buffer) for the Muon matrices. That keeps BF16
   storage and removes the rounding random walk.
3. Restart from the base weights (or from the 0.25 shrink) with fresh optimizer
   moments. The current moments and most of the current delta are noise. Keep the
   exposure history as history.
4. Normalize per decision rather than per invocation, make the within-decision
   mass token-aware, and fix the family-scale cap.
5. Monitor coherence: displacement versus root-sum-of-squares of windows, and
   the exponent-bucket SR fit, at each saved checkpoint. A ratio near 1 means
   the run is diffusing.

## Workspace and execution guidance

Read `AGENTS.md`. Root main at document preparation: `e14dd0476`; unpushed work
exists. Preserve unowned WIP in `scripts/coreweave/prebuilt.py`,
`scripts/neuralese_data/teacher_graph_student_twin.py`,
`scripts/neuralese_data/check_vllm_lfm3_parity.py`, `scripts/vllm_fork_wheel.py`.
Use an isolated worktree for independent changes and coordinate before GPU use,
learner changes, source mutation, cleanup or restarting shared services.
Do not restart teacher generation for this LFM3 recipe.

Python: `/home/werg/natlang/.venv-neuralese/bin/python`.
CoreWeave SDK venv: `/home/werg/.local/share/natlang/cw-venv/bin/python`.
Node24: `/home/werg/.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin`.
Set `PYTHONPATH` to the intended checkout and its `training/neuralese` directory,
and `NATLANG_REPO` to that checkout; use a disposable `NATLANG_HASH_CACHE`.
All DGX tests/builds/probes run under `scripts/memory_ledger.py` with a measured
budget. Large cached-vs-graph qualifications need ~55 GiB CUDA; do not run them
unbudgeted or beside another large GPU consumer.

Frozen full check is running in
`/home/werg/data/dgxops/wt-main-check-b37c417a6`, ledger
`natlang-main-frozen-b37c-1029`, log
`/home/werg/data/dgxops/check-b37c417a6-full.log`.
Its3064 plain Python tests passed; native suite was still running (~68%) when
last checked. It tests b37 plus a recorded local-WIP overlay, **not** all subsequent
XML/composition/convolution changes. Do not call latest main fully green from it.
The overlay hashes are in `check-b37c417a6-local-overlay.json` in the same ops
directory. Focused integrated XML checks passed separately as recorded above.

Nothing in this brief authorizes discarding valuable data, weakening identity
checks, changing protected splits, relaxing gradient tolerances to pass, or
treating partial exposure as production completion. Current AGENTS/owner
instructions and actual evidence remain authoritative.
