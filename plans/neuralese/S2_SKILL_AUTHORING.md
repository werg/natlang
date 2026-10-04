# S2: Crisp skill authoring

Draft, 2026-10-03. Detailed plan for stage S2 of the [Neuralese programme](README.md). It follows the decisions table in the README and [S0 revision 2](S0_SPEC.md), especially §7 (context as a curried argument), §7.3 (executable nodes come from files; data is open), §8 (termination) and §9 (learning surface). Proposals the owner has not yet decided are marked **Proposed** and listed in §12.

S2 is continuous. It starts immediately on the current runtime and keeps producing data and checkpoints for the whole programme. Its purpose is twofold: a crisp self-improver that creates, revises, selects and repairs reusable skills, judged on separate later tasks; and the episodes, trajectories and evaluation protocol that S6 turns into soft skills and learned updaters.

## 1. Starting point

What exists today:

| Component | Location | State |
| --- | --- | --- |
| Authored improver application | `applications/program-improver/` (`improveStep.nl`, `componentSearchStep.nl`, `counterexampleStep.nl`, `reducers/`) | Directory-reducer steps over a target folder, driven by `folder.iterateOn` with `propose`/`evaluate`/`accept` ([NATLANG_PROGRAM_IMPROVEMENT.md](../NATLANG_PROGRAM_IMPROVEMENT.md) §3). Edits instructions, guidance and structure of the current task's program. |
| Improvement host | `ts-host/src/improvement/` | Program repository, evidence reader, evaluator, journal, counterexamples, portfolios, transformations. |
| Search engine | `ts-host/src/optimization/` | GEPA engine vendored from Ax, reflection strategy, checkpoints, promotion. |
| Adaptation artifacts | `ts-host/src/adaptation/` | `natlang.adaptation/v1` with component kinds `lambda.instructions` and `program.guidance`. |
| Optimizer curriculum | `ts-host/scripts/self-improvement/` | ~55 scripts; `build-training-slate.mjs` produces a 192-task slate (16 families × 12 variants, `data/teacher/self-improvement/task-slate-v1/`); `publish-optimizer-training.mjs` (commit 187f1cc) and the default-producer change (be6a5e4) publish optimizer trajectories into the general SFT corpus. |

What is missing: skills as first-class assets, any reuse across tasks, support/query episodes, and evaluation of an improvement on cases other than the ones it was made from. Today's improver optimises the program in front of it.

## 2. What a skill is

A **skill** is what it is in agent harnesses: a folder with a `SKILL.md` (YAML frontmatter with `name` and `description`, then markdown instructions) and optional supporting files — scripts and code, references, assets. Natlang skills start out as exactly these standard markdown-and-code skills and stay compatible with them: a standard skill folder loads unchanged, and a natlang skill remains readable by other harnesses. What natlang adds is that skills are bound into contexts (S0 §7), may carry typed metadata, and can become soft.

Under S0 §7.3 the contents of a skill fall into the two categories that matter:

| Contents | Category | Can be added at will? |
| --- | --- | --- |
| `SKILL.md`, references, examples, assets, typed knowledge, `.nz` data | Data | Yes. Data entries are open; a skill can gain files at any time. |
| Callable code: `.nl` functions and TypeScript modules the skill offers as helpers | Executable nodes | Only by being written to files in a staged tree by a directory reducer, compiled and loaded (S0 §7.3). Once they exist they can be edited within their interfaces. |

Scripts meant to be run by a harness's shell are not executable nodes of the natlang program; natlang code calls skill code only through `.nl` and TypeScript helpers.

### 2.1 Layout

```text
skills/
  sql-join-keys/
    SKILL.md          # standard frontmatter + markdown instructions
    references/       # optional further documents, read on demand
    examples/         # optional worked examples
    helpers/          # optional executable nodes: .nl and .ts files with their callable folders
    tests/            # optional cases the skill must pass, with their checks
```

**Frontmatter.** The standard fields come first: `name` and `description`. The description says what the skill does and when to use it; it is the only thing a caller sees before loading the skill, so it carries what applicability information exists. Natlang adds an optional `natlang:` block:

| Field | Meaning |
| --- | --- |
| `scope` | Data injected automatically into the eval scope of a function that has the skill bound: named, typed bindings whose values come from files in the skill (typed JSON, `.nz` exports) or literals. They appear in the call's opening declarations like parameters, without the model having to load anything. |
| `exports` | Typed interface of the skill's helpers and scope bindings, in natlang type syntax. |
| `requires` | Other skills and services it depends on. A dependency on a service grants nothing; the binding function must already have that service. |
| `tests` | Test cases and the checks that decide them. |
| `provenance` | Episode, support cases, author (teacher or student, model and revision), parent skill revision. |

**Soft skills.** A soft skill is a `.nz` file (S0 §5) or a folder containing one. Where a markdown skill keeps its frontmatter in YAML, a `.nz` skill keeps the same fields (`name`, `description`, `natlang:` block) in its safetensors metadata, and its body is a `Neuralese` export instead of markdown. Mixed skills are ordinary: a folder can hold `SKILL.md` and `.nz` files side by side, and `scope` can inject Neuralese values from them.

**Semantic discovery metadata.** `description` and optional `summary` have type `string | Neuralese<string>` (including dialect-qualified forms). For soft skills the normal representation is a Neuralese value; the description is an optimisable soft discovery/applicability signal, alongside the soft body. IDs, names, revisions, paths, export names, requirements and interfaces remain crisp. `.nz` metadata stores the canonical `$neuralese` reference to the semantic block; the block must be persisted with the asset and loaded into the runtime store. Discovery listings carry that reference through the existing Neuralese content-part transport before body selection. There is no automatic gloss, implicit text fallback, or eager body read. Markdown skills retain text descriptions and can also explicitly carry a Neuralese reference in their YAML/JSON frontmatter. Text remains an explicit available representation for soft assets, but future soft-skill authoring should create soft descriptions by default.


### 2.2 Loading

Loading follows the standard progressive-disclosure pattern of agent harnesses:

1. A function's call opening lists each bound skill by `name` and `description`.
2. The model loads a skill's body when it judges it relevant (`read_code("skills.<name>")` for markdown, which renders `.nz` bodies as literals through the read port), and reads references and examples on demand.
3. Helpers are callable like any callable item.
4. In addition, each bound skill's `scope` bindings are injected automatically into the eval scope and listed in the opening's declarations, with Neuralese values rendered as literals.

Opening size grows with the number of bound skills and their declared scope, not with the length of their bodies.

### 2.3 Where skills live

Skills are part of a function's context and are bound with it, so the set of skills a function sees is fixed for that binding and is what optimisation acts on:

- **Per program, implicit by default.** A program's context includes its skills folder; its functions see those skills without explicit selection, as companion-folder items are seen today. Explicit selection is rebinding (S0 §7.2).
- **A global mutable pool at the top level.** The host keeps a skill pool that programs draw from when they are created or rebound. Changing the pool produces a new pool revision; it does not change any already-bound context. A program picks up pool changes only when it is rebound, so optimisation always works against a fixed context, and promotion can publish an improved skill back to the pool.
- There is no shared library that changes underneath running or optimised functions.

### 2.4 Skills are crisp until S5

In S2 every skill is crisp: markdown, typed data and code. S5 and S6 give skills soft forms (§9). The layout does not change; `.nz` files join or replace the markdown body.

## 3. Self-improvement as context → context

Self-improvement becomes a function from a context and evidence to a new context (S0 §7.2). The current improver already has this shape in folder terms; S2 makes it the general mechanism and adds skills.

- **Steps are directory reducers over staged trees.** A step works on a private copy, writes skill folders and edits existing entries, and returns typed state. `folder.propose` gives a speculative revision with its diff; `folder.accept` installs it. These are the existing semantics of [NATLANG_PROGRAM_IMPROVEMENT.md](../NATLANG_PROGRAM_IMPROVEMENT.md) §3.3.
- **New helpers are written, compiled and loaded.** A proposal that adds helper files is compiled before it can be evaluated. Compile failure is an ordinary rejected proposal with its diagnostics.
- **Selection is rebinding.** Giving a function a skill set is `fn.in(base ∪ selected)` (S0 §7.2). Until the S4 runtime implements `in`, selection is implemented by writing the selected skill folders into the staged tree, which is observably the same.
- **Promotion** binds the program to the accepted revision (`folder.apply` for real directories).
- **Termination.** Improver loops use TypeScript `until` predicates and therefore carry `withLimit` or `withMeasure` (S0 §8). Natural-language stopping predicates, where used, run under the dedicated predicate prompt. S0 §8 supersedes the termination section of NATLANG_PROGRAM_IMPROVEMENT.md §3.5.

### 3.1 Migrating today's engine

| Today | After S2 |
| --- | --- |
| `lambda.instructions` and `program.guidance` overlays in `natlang.adaptation/v1` | Ordinary edits of context entries: instructions are edits of an executable node within its interface; guidance is a data entry. The adaptation artifact remains as an export format for a context diff, with a migration entry. |
| GEPA search over components | Search over context revisions. Candidates may include skill folders. The exact helpers (selection, Pareto bookkeeping, statistics) stay TypeScript. |
| Improver optimises the current task | Improver optimises a skill library against support cases and is judged on query cases (§4). |
| Optimizer curriculum slate | Kept as a family source. Its tasks become episode material (§5.1). |

Every API change gets its training-data migration entry under `training/api-migrations/`, following the existing rule.

## 4. Episodes and the evaluation protocol

### 4.1 Episode record

**Proposed** schema `natlang.skill-episode/1`, model-neutral like `natlang.program/2`:

| Field | Meaning |
| --- | --- |
| `id`, `family`, `split`, `source_groups`, `license` | Identity and lineage, with split groups closed as in S1. |
| `target` | The program (a `natlang.program/2` root and context) whose behaviour the skills should improve. |
| `library` | The starting skill library: empty, existing, or deliberately corrupted (§5.3). |
| `support` | Cases the author may see: tasks, their runs with the starting context, outcomes, checks and failures. |
| `query` | Separate cases used only for evaluation: same family, disjoint source groups. Never visible to the author. |
| `transfer` | Optional cases from a related family, also never visible, used to measure transfer. |
| `operations` | Which authoring operations are allowed (§4.3). |
| `limits` | Step limit for the improver's iteration. |

An episode's result records the authoring trajectory, the accepted context diff, test results, and the paired evaluation in §4.2.

### 4.2 Protocol

1. Run the target with the starting context on support cases. Record outcomes and traces as evidence.
2. Run the author (teacher or student improver) on the support evidence. It produces a revised context.
3. The host evaluates the starting and revised contexts on query cases with identical seeds, in fresh workers, through a host-issued evaluation ticket. The author never sees query inputs, expected values, traces or per-case scores.
4. Record paired query gain, transfer gain, skill test results, and per-skill attribution by leave-one-skill-out ablation.

### 4.3 Leakage rules

- Query and transfer cases are disjoint from support by source group, not only by instance. Near-duplicates across the boundary are removed (S1 deduplication).
- The author's evidence comes only from support runs. Aggregate query results may be revealed only after the episode ends.
- A query task's solution, expected value or later trajectory never appears in support, in a skill's examples, or in a skill's tests.
- Skills authored in one episode enter a later episode's starting library only if that later episode's query cases are disjoint from every support case the skill was made from.
- All attempts at one episode stay together in one split.

### 4.4 Authoring operations

| Operation | Meaning | Trigger in support evidence |
| --- | --- | --- |
| Create | Write a new skill folder (data, or with helpers via the staged tree) | Recurring failure that reusable knowledge or a procedure would fix |
| Revise | Edit an existing skill's description, summary, procedure, knowledge, examples or helpers within their interfaces | Skill is applied but partly wrong |
| Select | Choose which library skills a function is bound to (rebinding) | Many available skills, few relevant |
| Repair: missing | Notice that a needed skill is absent; create it | Failure with no applicable skill |
| Repair: irrelevant | Remove a skill from a binding | Skill is bound but distracts |
| Repair: incorrect | Revise or retire a skill that causes errors | Ablation shows the skill hurts |
| Retire | Remove a skill from the library | Superseded or consistently harmful |
| Test | Write or extend a skill's tests | Every create and revise |

### 4.5 Discovery and use optimisation

Self-improvement includes tuning descriptions and summaries to improve task-dependent skill discovery, for crisp and soft skills alike. A metadata-only change is a valid candidate and must be evaluated as such. Crisp descriptions can be rewritten from support evidence; Neuralese descriptions are trainable values in the same optimisation context as soft bodies. The objective is the task's necessary and helpful skills: avoid missed useful skills, irrelevant reads/applications, and redundant use without sacrificing quality. Binding/discoverability, reading a body, and applying instructions or calling a helper are distinct events; merely listing a skill is not evidence of its use.

Do not define a universal target skill count or penalise every additional skill. There can be multiple equally good skill combinations and interactions between skills. Evaluate metadata-only and body-only variants, disabled-skill ablations, and distractor-rich support/query episodes. Judge appropriateness through independent task quality and measured contributions, not through a description matching a hidden label. Track body reads, helper calls, costs and ablation effects separately. Missing/unnecessary-use labels require evidence; a body read alone cannot prove cognitive application or necessity. Prefer a less costly sufficient configuration when quality is preserved. Keep all tuning/selection on support; query and transfer remain sealed, with no adaptive query-based tuning.

Current crisp authoring permits description/summary edits in SKILL.md and now explicitly prompts this objective. Actual discovery-event accounting, attribution ablations, metadata-only campaign cases, distractor libraries, and soft-description gradient optimisation remain to be wired into collection/evaluation. Existing quality gates do not yet certify optimal skill selection.

Selection is itself a decision the model makes (which skills apply), recorded as a discrete choice in the trace (S0 §11.2).

## 5. Corpus generation

### 5.1 Sources of episode material

| Source | Episode use |
| --- | --- |
| S1 Natlang task adapters (SQL, function calling, code, offline web, repository, reasoning, retired natlang sources, worlds, SWE) | Families of executable tasks with checked results. Support and query cases are drawn from the same family. |
| Schnitzeljagd cross-experience episodes (S1 §9) | Ready-made support/next-task pairs grouped by repository or tool schema. |
| Optimizer curriculum slate (192 tasks, 16 families) | Families with gold references; extended with query variants. |
| Natlang's own failure index (`index-failures.mjs`, `LEARNING_LESSONS.md`) | Failure clusters as repair-episode triggers. |
| Directory campaign (`build-directory-campaign.mjs`) | Folder-editing tasks as skill targets. |

### 5.1a Graded families

Self-improvement needs a loss landscape: the host scores every answer on a continuous scale, independently of any model, so a skill edit shows as a measurable gain or loss on sealed query cases. Exact-match families give only right/wrong. Graded families in use or planned:

| Family | Graded quality | Status |
| --- | --- | --- |
| Combinatorial optimization: knapsack, bin packing, weighted tardiness, graph coloring, TSP | Normalized gap between exact host-computed bounds (`ts-host/src/skills/objective.ts`) | Built: `build-optimization-episodes.mjs`, replicas × starting-library variants (empty, distractor, misdescribed, incorrect, redundant) |
| Text-to-SQL (Spider) | Result-set F1 of the returned query against the gold query, both executed read-only (`ts-host/src/skills/graded.ts`) | Built: `build-sql-episodes.mjs`; one database per episode, distinct-gold-SQL support/query, transfer to another database, dev databases held out |
| Code (KodCode) | Fraction of reference unit tests passed in a network-less pinned Python sandbox | Built: `scripts/build_code_skill_episodes.py`; one KodCode subset per family, moderate-difficulty band, benchmark-near problems excluded |
| Multi-hop QA (HotpotQA distractor) | Answer token F1 | Built: `scripts/build_graded_skill_episodes.py --source hotpot-answer`; families by question type × level |
| Retrieval and ranking (HotpotQA supporting paragraphs) | Binary-relevance NDCG of the returned title ranking | Built: `--source hotpot-support` |
| Logic (knights and knaves) | Fraction of inhabitants classified correctly | Built: `--source knights`; families by number of inhabitants |
| Tool calling (xLAM 60k) | Multiset F1 over call names and argument bindings | Built: `--source xlam`; families by call shape and tool choice |
| Probabilistic classification | Brier or log score of predicted probabilities | Planned |
| Interactive environments (TextWorld) | Score and steps to goal | Planned |

All scorers go through one registry (`ts-host/src/skills/scoring.ts`), used by both the collector and the offline exporter, so a graded episode replays exactly.

### 5.2 Teacher collection

- Teachers run the authored improver application itself, not a host-side optimiser: the local Qwen3.6 on the DGX and the paid teachers (Luna, Bunny). The collector journals every reply, as `teacher-collector.mjs` does today.
- Several attempts per episode with different seeds and teachers. All attempts are kept.
- Every attempt is executed: skills' tests are run, query evaluation is run, compile errors are recorded.

### 5.3 Constructed repair episodes

Repair needs libraries that are wrong in known ways. From admitted successful skills, construct:

- **Missing:** remove a skill that query evaluation showed to be necessary.
- **Irrelevant:** bind a plausible skill from another family.
- **Incorrect:** corrupt a skill (wrong step in a procedure, wrong value in knowledge, a bug in a helper) and check that it lowers query performance.

The corruption is recorded as the episode's label, so repair can be scored exactly.

### 5.4 Admission and labels

| Outcome | Use |
| --- | --- |
| Positive query gain, tests pass, no regression on unaffected cases | Success: SFT target for the authoring trajectory |
| No gain or regression | Labelled failure: negative of a preference pair against a successful attempt on the same episode, and a contrastive example |
| Compile failure, invalid edit, policy violation | Labelled failure with the diagnostic; repair continuation if a later attempt fixed it |
| Correct repair of a constructed defect | Success, with the defect label |

Failed attempts are never treated as demonstrations. Admission rules follow the existing lineage and migration discipline (`plans/DATA_LINEAGE.md`, `training/data_sources.json`).

## 6. Coverage and learning curves

- **Coverage matrix:** family × operation × skill kind (data, helper) × starting library (empty, existing, missing, irrelevant, incorrect) × outcome. Generation targets every cell; the report lists empty and thin cells.
- **Learning curves:** after each student round, plot held-out authoring success and query gain against the number of admitted episodes per cell. Generation continues in cells whose curves still rise; it is rebalanced away from flat cells.
- Raw row counts are reported but are not a target.

## 7. Student training

- Episodes are rendered through the existing pipeline (collector → materializer → renderer) with the S0 §11.4 rewrite passes (eager typing, explicit captures). Every model decision point in the improver is a training example: diagnosis, operation choice, skill writing, test writing, selection, acceptance.
- SFT on successful trajectories; preference training (`scripts/train_dpo.py`) on paired successes and labelled failures from the same episode.
- The authoring corpus joins the general crisp corpus with a share set per round, and ordinary crisp data stays in the mix to protect execution ability.
- Each round produces a crisp checkpoint. S3 and S5 rebase onto later crisp checkpoints selected by execution evaluation.

## 8. Evaluation and exit

### 8.1 Held-out sets

- **Instance held-out:** new episodes from training families, disjoint by source group.
- **Family held-out:** families never used in training, for transfer of the authoring ability itself.
- **Reuse:** skills authored in one episode applied to later, disjoint tasks of the same family without further authoring.

### 8.2 Metrics

| Metric | Definition |
| --- | --- |
| Authoring success | Share of episodes where the student's revised context improves query success over the starting context, with tests passing and no regression on unaffected cases |
| Query gain | Paired mean improvement in query success |
| Transfer gain | Paired improvement on `transfer` cases |
| Reuse gain | Improvement from reusing an authored skill on later disjoint tasks |
| Repair accuracy | Correct identification and fix of constructed missing, irrelevant and incorrect skills |
| Selection quality | Precision and recall of selected skills against leave-one-out attribution |
| Valid-edit rate | Share of proposals that compile and respect policy |
| Teacher ratio | Student authoring success divided by teacher authoring success on the same episodes |

### 8.3 Evaluation rubric

There are no fixed exit thresholds. Evaluation runs continuously on the held-out sets, and the results decide what to improve next: more data for weak families, new repair constructions, training changes, or handing the current ability to S5 and S6. Each evaluation round is reviewed against this rubric:

| Dimension | Questions the review answers |
| --- | --- |
| Does authoring help? | Do revised contexts improve query success over the starting context, across families, and are the gains consistent or concentrated in a few? |
| Does it transfer? | Do gains hold on transfer cases, on held-out families, and when an authored skill is reused later without further authoring? |
| Is it safe? | How often do edits regress unaffected cases, fail to compile, or violate policy? |
| Can it repair? | Does it find and fix missing, irrelevant and incorrect skills, and which defect kinds remain hard? |
| Does it select well? | Are selected skills the ones that matter by leave-one-out attribution? |
| How close to the teacher? | Where does the student trail the teacher on the same episodes, and is the gap closing with more data? |
| What do failures look like? | A sample of failed episodes, read and categorised, with the categories tracked over time. |

The authoring ability and corpus are handed to S5 and S6 when the review judges that authoring helps and transfers on held-out families; generation and training continue afterwards.

## 9. Handoffs

- **S5 (program-level training):** data skills are context items with text. They are the first candidates for soft forms, initialised from their text (token embeddings) or written by the writer, stored as `.nz` next to the text. Helpers stay crisp.
- **S6 (soft skills and meta-learning):**
  - Episodes are exactly S6's support/query episodes. The protocol in §4.2 is reused unchanged, with soft revisions in place of crisp ones.
  - Crisp authoring is a step function from context to context. S6's learned updaters are step functions over soft values (S0 §9.6) and are warm-started from S2's authoring trajectories.
  - The gradient-tuning baseline (`valueAndGrad` over a skill's soft value with an optimiser in `iterateOn`) runs on the same episodes, so crisp authoring, gradient tuning and learned updates are compared at matched evaluation.
- **S7 (RL):** authoring and selection become policy-learning tasks, with query gain as reward.

## 10. Dependencies

- S0: skill semantics rest on S0 §7 (contexts, rebinding, executable nodes from files). S2 starts before the S4 runtime implements `in` by writing selected skills into staged trees, and switches to rebinding when S4 lands.
- S1: Natlang task adapters and environments supply families; their split groups and deduplication apply to episodes.
- S4: context values, rebinding and the context-interface check.

## 11. Work items

In dependency order:

1. Write the skill format (standard `SKILL.md` plus the `natlang:` frontmatter block and `.nz` metadata equivalent), loading with scope injection, and the per-program/global-pool model (§2), as an extension of the S0 context chapter.
2. Write the `natlang.skill-episode/1` schema and the host evaluation ticket for query and transfer cases (§4).
3. Implement leakage checks: source-group disjointness, near-duplicate removal, skill-provenance checks for later libraries (§4.3).
4. Extend the improver application with skill operations: create, revise, select, repair, retire, test, as directory-reducer steps (§4.4).
5. Episode builders from the optimizer slate, failure index and directory campaign (available now); then from S1 adapters and Schnitzeljagd episodes as they arrive.
6. The defect constructor for repair episodes (§5.3).
7. Teacher collection on the DGX and with paid teachers; admission and labelling (§5.2, §5.4); API-migration entry.
8. Coverage matrix and learning-curve reports (§6).
9. Student rounds: SFT and preference training, crisp checkpoint per round (§7).
10. Held-out suites and the exit report (§8).
11. Migrate the improvement engine and adaptation artifacts to context revisions, then to `in` rebinding when S4 provides it (§3.1).
12. Package episodes, trajectories and the evaluation harness for S5, S6 and S7 (§9).

## 12. Decisions on former open questions

Resolved by the owner on 2026-10-03:

1. **Skill format:** standard markdown-and-code agent skills (`SKILL.md` with `name`/`description` frontmatter plus supporting files), extended by an optional `natlang:` frontmatter block; `.nz` skills keep the same fields in their safetensors metadata (§2.1).
2. **Loading:** standard progressive disclosure, plus automatic injection of the scope data declared in metadata (§2.2).
3. **Libraries:** skills are part of the context bound to a function; implicit per-program selection by default; a global mutable pool at the top level, which affects programs only when they are rebound (§2.3).
4. **Applicability:** no separate field; the standard `description` says when to use the skill.
5. **Exit:** no fixed thresholds; continuous evaluation reviewed against the rubric in §8.3.

### 2026-10-04 Pop authoring progress and resource interruption

Pop-owned runs use `pop-*` namespaces and isolated Git worktrees on the DGX;
source synchronization is frequent fetch/merge/push of main. The main DGX checkout
belongs to concurrent development and must not be overwritten by artifact sync.

The expanded optimization and corrected selection packets contain 80 train and
24 protected validation episodes, 1296 cases and 916 role/split-consistent groups.
The provider-free combined audit recomputed 672 optimization bounds, with no
schema, source-group, input-alias boundary, or reference errors. This is packet
integrity evidence, not evidence of actual model improvement. Collection starts
with three optimization families before scaling the train partition.

Optional bounded host-only ablations now compare selected skills with restored
baseline descriptions, restored bodies and removal of a skill. Raw paired effects
and content-free observed skill events remain sealed from the author and excluded
from SFT. Metadata-only campaigns freeze bodies and all metadata except description
and summary. Forty-three focused authoring/replay/ablation tests passed; a further
sealed-event test confirms no test inputs, model trace or evidence-reader access
is exposed to the author. The negative original interrupted export has zero SFT
rows and seven negative evidence records, with no provider calls or invented DPO.

At 06:16 UTC the DGX suffered a **global** OOM event, killing both Pop pilot-v2
and general Qwen generation. The pilot's 8 GiB MemoryMax is not evidence that it
hit that cgroup limit: kernel records report `global_oom`. V2 interrupted artifacts
are preserved and are not model-quality failures. Available RAM recovered after
process deaths; Qwen server survived but had zero active requests. Restart Pop
pilot in a fresh namespace at lower parallelism; monitor whole-machine unified
memory rather than increasing the cgroup cap. Do not modify another agent's runs.

Remaining: obtain actual complete query/transfer comparisons, replay positives
exactly offline and admit only verified support SFT, then scale the 80 train
configuration episodes with descriptive improvement rates and rejection review.
Full soft-body loading and gradient-based soft description tuning remain separate.
