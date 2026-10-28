# Learned optimizer: decomposition, part by part

Status: the Natlang program of [plans/neuralese/JOINT_LEARNED_OPTIMIZER.md](../../plans/neuralese/JOINT_LEARNED_OPTIMIZER.md).
Its crisp mechanism (bank, writer, mixture head, composition, fits, objectives, trainers) is
[`training/neuralese/natlang_neuralese/tune/learned/`](../../training/neuralese/natlang_neuralese/tune/learned/DECOMPOSITION.md).
Every natural-language function runs on the executor's own base (LFM2.5-350M) with the optimizer's adapters; the
self-hosted teacher is only a measured fallback (plan "Everything on the 350M").

Decisions: **fn** (its own natural-language function), **inline** (one instruction of another function),
**crisp** (exact helper), **service** (host capability), **pluggable** (crisp and natural-language
implementations behind one interface; the setting selects).

| Part | Decision | Where | Why |
| --- | --- | --- | --- |
| One adaptation episode | fn (root) | `adapt.nl` | Orchestrates analysis, revision and selection; `iterateOn` with the measure `remainingSteps` gives structural termination |
| One episode step | fn | `adapt/adaptStep.nl` | Revises the complete prompt/skill/LoRA collection jointly every iteration |
| Interpret evidence: focus, key evidence, digests | fn | `adapt/interpretEvidence.nl` | Semantic judgement over exact failure facts; `digest` is a native producer span the writer consumes |
| Choose focus | fn | `adapt/chooseFocus.nl` | Its own stage so tuning has exact hindsight labels (`tune/learned/meta.py`); the meta-learner is tuned like any program |
| Choose revision operations per binding | fn | `adapt/chooseRevision.nl` | Policy over contracts, dependencies and failures; the services change exactly the plan's selection (`apply.Context.selection`) |
| Choose operator (learned proposal, retrieval, local refinement, keep) | pluggable fn | `adapt/chooseOperator.nl` | The memetic strategy learns; the crisp baseline is `apply.episode` |
| Stop | fn behind the structural measure | `adapt/shouldStop.nl` | Benefit/cost judgement; the host measures cost exactly |
| Exact evidence facts and permitted view | service `optimizer.evidence` | `evidence.py` | Causal visibility and failure facts are exact |
| Write states and LoRA mixtures | service `optimizer.propose` | `writer.py`, `mixture.py`, `episode.py` | The native writer and the component-token mixture head (distribution over bank components plus null, group gains); composition and validation are exact |
| Score a bundle on support evidence | service `optimizer.evaluate` | `apply.Context.evaluate` | Exact loss; no model judges it |
| Local in-bank refinement | service `optimizer.refine` | `fit.py` | Gradient operator inside the loop |
| Gradient-free retrieval over the bank | service `optimizer.retrieve` | `apply.Context.retrieve` | The LoraHub-style baseline, same interface |
| Release fragment | service `optimizer.release` | `apply.release_fragment` | Atomic, content-addressed |
| Outer training controller | fn (root) | `train.nl` | `iterateOn` over training cycles with a finite cycle measure |
| Select training work and loss mixture | pluggable fn | `train/scheduleTraining.nl` | Operations decision from recorded metrics; quotas and receipts stay crisp |
| One training cycle | fn | `train/trainStep.nl` | Calls exact trainer services in the scheduled order |
| Supervised/policy updates, refits, publication, ladder | service `trainer.*` | `trainer.py`, `fit.py`, `ladder.py` | Gradients, optimizers and measurements are exact |
| Task outcomes of sampled proposals | service `trainer.collectProposals` with `outcomes` | `__main__.proposals`, `natlang tune --score-proposals` | Proposals run as releases on their programs' query roots; scores are the programs' own |
| Service process | crisp | `python.mjs` (`PythonService`, `AdaptationServices`), `apply.serve` | One persistent process keeps contexts loaded for the program, its training and its tune services |
| Tuning this program | crisp | `tune.services.mjs`, `tune.reward.mjs` | Whole `adapt` roots reach the live services; each stage and episode is graded by measured support-loss improvement |

## Model-facing text

The `.nl` bodies, the optimizer request rendering (`request.py`) and evidence rows (`evidence.compact_rows`) are
model-facing and are recorded in `plans/MODEL_FACING_CHANGES.md` until measured live on the 350M executor.
