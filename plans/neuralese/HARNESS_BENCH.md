# The harness bench: teaching a small Neuralese student to work like a strong agent, with harness support

Proposal, 2026-10-09, from the owner's direction. The pi port (applications/pi) with its companion
(applications/pi/COMPANION.md) is also a training bench for Neuralese. Take agentic trajectories that strong agents
produced. Put our small, Neuralese-capable model in the agent's role. Then learn the harness support (background
knowledge, research, context economy, delivered through Neuralese) that brings the student closest to the teacher's
behavior and success.

The student does not have to become as capable as the teacher. The harness carries part of the work: what the
teacher figured out by exploring, the companion finds and hands over densely. The question the bench answers is how
far learned harness support closes the gap, and at what context cost.

All training follows the shared recipe (TRAINING_RECIPE.md): foundation gates first, runtime qualification for the
exact weights, compression only as explicit qualified operators. Data enters through the corpus registry
(`training/neuralese_corpora.json`) with immutable manifests, and admission is a separate, explicit decision.

## 1. Pipeline

```
teacher trajectories ─► normalize to pi transcripts ─► reconstruct environment states per step
   ─► run the companion offline on each prefix (real research on the real state)
   ─► score offers with hindsight (what the teacher later needed)
   ─► rewrite trajectories: injected offers, shaped outputs, optional shortcuts
   ─► train: student on teacher actions given offers; writer of Neuralese offers through the student's loss
   ─► evaluate live: student + companion in pi, on held-out tasks
```

### 1.1 Sources

- Public coding-agent trajectories with executable environments: SWE-bench-family tasks, which come with a
  repository, a commit, an environment and tests. Candidate releases include SWE-Gym, SWE-smith and Nebius'
  SWE-agent trajectory sets. Availability and licence are checked at registration and recorded per record
  (programme decision: restrictive licences are acceptable, and the licence is always recorded).
- Our own teachers through the pi port: the local Qwen3.6 on the DGX and the paid teachers (Luna, Bunny), running in pi
  on eval tasks and SWE environments. These trajectories are already in pi's transcript form, and every step is
  recorded in the call store.
- Natlang program trajectories (the existing teacher collector) are a related source for the same student. They are
  not in scope here.

### 1.2 Normalize: harness dialects

Teacher agents use different tool sets: SWE-agent's ACI commands, OpenHands' `str_replace_editor` and
`execute_bash`, other agents' `find_file` and `search_dir`. Each source gets a **harness dialect**:

- a mapping from the source's tool calls and observations to pi transcript entries;
- where a mapping would lose information, a pi extension that registers tools with the source's own semantics (pi
  tools are ordinary extensions), so the replayed harness matches the trajectory instead of distorting it. This is
  the owner's point about refactoring tools to form harness versions that match the trajectory;
- model- and harness-specific conventions stripped, content and supervision kept (programme decision).

The dialect is recorded per record. The student is trained across dialects, so it learns tool use as an interface,
not one harness's habits.

### 1.3 Reconstruct the environment at every step

Replaying a trajectory's tool calls in its environment rebuilds the workspace state before each step. A replayed
observation is compared with the recorded one: equal steps are *verified*, and divergent steps are marked (for
example nondeterministic output or timestamps). This gives real states to run the companion on, rather than
imagined ones.

Trajectories without a reproducible environment are still usable, with lower trust. Their states are approximated
from the recorded observations (files the agent printed), and the record says so.

### 1.4 Generate the background information: run the real companion, offline

This addresses the owner's open problem: how to fabricate the background information the harness injects. Mostly,
we do not fabricate it. On each reconstructed prefix state, the bench runs the actual companion (its indexing,
research, compression and critique functions) with only what a live companion would have had at that point: the
transcript prefix, the workspace at that step, and the allowed documentation sources. Its executor can be a strong
model (Qwen3.6 or a paid teacher), because offline research has no latency constraint.

The result is a set of candidate offers per step (briefing items, shaped outputs, retrieved facts), each produced by a
causal process that the live companion can repeat.

### 1.5 Hindsight: which offers mattered

The rest of the trajectory shows what the teacher needed: files it read later, the symbols it searched for, the
location it edited, the test output that changed its mind, the final patch. Hindsight is used to:

- **score** each offer: did it contain information the teacher later spent steps to find? How many steps (tokens,
  tool calls) would it have saved?
- **select** a small set of offers per step under a context budget (a knapsack over estimated savings);
- **supervise the companion's planner**: the hindsight-useful information becomes the target for "what should I
  research next" (hindsight relabeling, as in HER).

Hindsight never writes the offers' content. An *oracle* variant (future information injected directly) is built
only as a labelled upper bound for evaluation, never as training input for the companion's writer. A leakage audit
checks every injected offer against the information available in its prefix state. An offer whose content appears
only in the future is rejected.

### 1.6 Rewrite trajectories

For each source trajectory, the bench writes one or more harnessed versions:

- **Augmented:** the same teacher actions, with the selected offers injected where the companion would have delivered
  them (the briefing section, shaped tool results, recall handles).
- **Shortcut:** exploratory teacher steps that a delivered offer makes unnecessary are removed, when the remaining
  actions still replay to the same outcome in the environment (tests pass, same final patch). The student's target
  is then shorter than the teacher's trajectory, because the harness did that part of the work.
- **Compressed:** old observations are replaced by the companion's summaries plus recall handles, which models
  context economy.

Every rewrite is verified by replay where an environment exists. A rewrite that fails to reach the outcome is
dropped, and the failure is recorded.

## 2. Training

Two learners share the trajectories.

- **The student** (the Natlang/Neuralese line: LFM2.5-350M or Maple, whichever is current) learns the teacher's
  next action given the harnessed prefix. This is whole-trajectory supervision (owner decision): prompts, offers and
  tool output are trained at lower weight, and actions at full weight.
- **The companion's Neuralese writer** learns to encode offers as `Neuralese<Briefing>` and `Neuralese<Fact>`
  blocks. Its signal is the student's loss on the teacher's following actions, through the read port (graph replay,
  S0 §9). It learns what to encode and how densely, from whether the student then acts like the teacher.

Stages, each a recipe stage with its own gate (TRAINING_RECIPE.md):

1. **Text offers.** The student is trained on augmented trajectories with offers as text. This is the baseline and
   the evidence that harness support helps at all.
2. **Neuralese offers.** The same trajectories, with offers delivered as blocks from the qualified write and read
   path of the exact weights. Compare action match, context tokens and task success with stage 1 at equal budgets.
3. **Compression operators.** Neuralese summaries of old observations and of knowledge-base entries. These are
   explicit operators, qualified separately.
4. **Planner learning.** The companion's own natural-language functions (what to research, what to offer) learn from
   hindsight labels and usefulness statistics (LEARNING_CONTINUUM.md operators).
5. **Live and RL.** The student with the companion runs in pi on held-out tasks. Rewards come from tests (S7), credit
   flows to both the student's actions and the companion's offers.

## 3. Evaluation

On held-out tasks with environments, live in pi:

| Configuration | Purpose |
| --- | --- |
| teacher alone | reference |
| student alone | the gap |
| student + text companion | how much harness support helps |
| student + Neuralese companion | the same at a lower context cost |
| student + oracle companion (labelled) | upper bound for the companion's information |
| teacher + companion | product value of the companion for strong agents |

Report task success, steps, context tokens, wall time and offer usefulness. Following the programme, there are no
fixed pass thresholds: reviews decide what to improve next.

## 4. Data handling

- Each source is registered with its licence, an immutable manifest and an explicit admission decision. Rewrites are
  derived corpora, with the source manifest, the companion's revision and executor, and the hindsight scorer's
  revision recorded.
- Environments (containers, repository snapshots) are listed in the manifest by digest, not copied into git.
- Storage goes on NVMe for anything the trainer reads hot. Bulk environments can live on the external disk.

## 5. Order of work

1. Companion skeleton in pi (COMPANION.md §7.2), measured live, because the bench injects what the live companion
   produces.
2. One source end to end on a small slice (about 50 SWE-bench-family trajectories with environments):
   normalization, replay verification, offline companion runs, hindsight scores, augmented and shortcut rewrites.
   Inspect the results, then register them as evidence only (no admission).
3. Text-offer training (stage 1) on the slice: does the student's action match improve over plain trajectories?
4. Scale sources. Then Neuralese offers once the channel is qualified for the student's weights.

## 6. Open questions

- Shortcut rewrites change what the student sees the teacher do. When the shortcut depends on an offer the live
  companion would not reliably produce, the student learns to over-trust the harness. Mitigation: shortcut only on
  offers whose reproduction rate is measured high (the live companion produces them again on rerun).
- Cross-tokenizer teachers: KL distillation needs matching tokenizers. Otherwise supervision is on actions (text)
  and decision readouts only (LEARNING_CONTINUUM.md).
- The cost of offline companion runs per step. Start with every step in the small slice. Then run only at step
  boundaries where hindsight shows information was gained.
