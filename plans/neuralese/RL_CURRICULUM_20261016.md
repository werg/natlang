# Executable task curriculum for both native students

Correct task execution earns reward. A task does not require Neuralese, a
communication edge or a minimum number of graph nodes. Where an episode has a
graph, the shared learner retains and updates its entire actual graph; task
selection never detaches a producer or substitutes a consumer-only trajectory.

## Shared scheduling

The typed policy is `applications/curriculum-advisor/selectTrainingTasks.nl`;
`rl/curriculum.py` is its crisp default adapter in the existing
`production.select_tasks` interface. Native `production configure` outputs now
include:

```json
"task_curriculum": {
  "schema": "natlang.rl-task-curriculum/1",
  "breadth_fraction": 0.25,
  "promotion_success_rate": 0.6
}
```

Task metadata is `curriculum.rl = {level, family, qualification}`. Lower levels
start practice; actual scored success moves practice upward. Existing tasks
without metadata remain eligible at level 3. Reserve at least one rotating row
and the configured fraction of every selection for complete breadth, including
solved tasks. This cursor eventually visits every available task regardless of
practice difficulty. New tasks enter the same rotation. Solved groups no longer
retire permanently, including in legacy configs without the adaptive setting.

The controller keeps the latest scored observation per task in
`task_curriculum_history`, with observed policy/round/attempt. This is scheduling
telemetry, never current-policy qualification or a substitute for captured
behavior probabilities. Unscored infrastructure does not update that history.
All-infrastructure rounds still demand repair. Zero-spread failures retain the
existing verified whole-trajectory RFT/seeding route. Mixed groups use GRPO.

An optional `select_tasks_command` receives JSON `{tasks, groupStats, budget,
curriculum}` on stdin and returns `[{row,samples}]`. This is the existing
pluggable boundary, not another scheduler. The crisp verifier checks identities,
distinct rows, declared allocation, uniform groups, and rotating breadth with
its last row last (the existing durable controller advances from that row).
Selection snapshots remain frozen through interrupted learning/publication;
policy and newly admitted tasks are reconsidered at the next round boundary.

Both participants use the same policy and native shared trainer. The production
builder `rl.production configure --native-state ...` supplies this setting;
existing deployed configs can add exactly the stanza above without replacing
model/optimizer/RNG/exposure or creating another training lineage. Existing
native_rounds/provider deployment and semantic-refresh boundaries are unchanged.

## Authored task inputs and live qualification

Rebuild the static packet with the existing compiled teacher runtime:

```sh
node scripts/neuralese_data/build_rl_task_curriculum.mjs \
  --runtime /home/werg/natlang/ts-host \
  --base-cases runs/decision-rich-train-20261005-v2/cases.jsonl \
  --out /home/werg/data/dgxops/rl-execution-curriculum-20261016
```

These are independently authored task inputs/oracles, not sampled training
trajectories. The base packet retains all 72 registered train tasks, SHA
`6f2a470a8de79d4fb9966fc5a5157ebdf16d3c58a02ae034322a2bdfcde20dca`.
The prepared qualification packet has 12 rows, SHA
`5839a9edd822d99abe003ca798b8636d8bf17cd18132bbf8a8903c3ebf5d735d`;
the pending full packet has 84 rows, SHA
`390fdd6d4094f5e2af3e08d045792d9c2ba6b37672ba252f436a8165a73eefe2`.
The builder records hashes and `training_admitted:false` in its manifest.

The four levels contain literal typed returns; arithmetic, sorting and record
access; compositional computation with optional tools; and independent typed
private writer/reader scaffolds. The three private authored roots deliberately
exercise writer-to-root return, writer-to-reader and two-writer-to-reader paths.
All exact expected outcomes use existing verifiers. Every private trajectory
contract sets minimum edges/nodes/iterations to zero: communication is exercised
by the diagnostic scaffold, never required by the reward.

For the 12-by-4 live window, retain the three private authored-root contracts.
Converting them to student-root mode removes the fixed communication exercise
and cannot qualify native-return transport. Production student-root collection
may use any correct solution; it has no mandatory private calls. A project
returning string can receive an intermediate `Neuralese<string>` from root eval
and continue through the existing ordinary write-out path; this has an existing
host test, but requires actual native executor qualification here.

**Pending:** root supervises baseline/candidate executor measurement using
retained trained step 6711 and an exact frozen runtime pair. That historical
student is a diagnostic actor, not the current cloud policy. About 48 samples
per variant and exact checkpoint/runtime/sampling/result evidence are required
in each intended root mode before adopting the new instructions. Authored-root
diagnostics do not qualify the corresponding whole-student-root prompts. The
production selector rejects
`qualification:pending-live-48`. Qualification is not corpus admission; publish
an explicit admitted full packet with the measurement binding and preserved base
rows before changing production cases. Until then, existing train tasks remain
usable with the corrected selector. Do not claim that pending rows were trained.

New successful teacher trajectories must come from the substantially stronger
self-hosted CoreWeave Qwen teacher. Student measurement outputs remain diagnostic.
Queue these task inputs for the next teacher window after live qualification;
retain complete verified trajectories, source/split/runtime/actor identities,
and transfer useful successes to both models through the shared RFT mechanism.
Protected held content, IDs and answers were not used to author this packet.

## Validation and limits

Focused Python tests exercise growing difficulty, all-solved regression visits,
new hard tasks, pending-source rejection, durable all-solved restart/appending,
and existing controller update/publication recovery. The real natlang host
contract accepts all nine ordinary scaffold oracles and rejects an incorrect
answer using a scripted provider. Static checks retain the three typed private
roots and their exact contracts. These checks validate mechanisms and task
oracles; they are not live model performance or completed RL training.

The historical 6750 held report was inspected only for diagnosis: 0/16, with
invalid actions, incomplete/budget and semantic failures, plus known native-ref
write-out debt. Those failures do not establish inability to learn or justify
training on protected evaluation content. Root retains runtime repair and
actual live outcome interpretation.
