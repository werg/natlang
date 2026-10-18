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

**Historical diagnostic stopped (reconciled 2026-10-18):** step7750 checkpoint
`f26338f6599334e59d6f30f9d27a63fa665dfb95d3d88972d6ea43fbfc2a5b94`
and frozen runtime
`e93c3ff248dfb73f84eae9f2baf1af3a7e4726ed3b23fd78ac6530c44a732613`
were diagnostic authorities, not current policy qualification. The actual
`terminal.json` under
`/mnt/external/natlang-development-data/native-curriculum-qualification-20261016/`
records an explicit stop at **2026-10-16 20:48:59 UTC**, return codes `[-2,-2]`.
Both preparations used the original `5839a9ed...` source, before its public-return
repair. Authored mode declared48 but scored only6 (3 successes,2 invalid actions,
1 semantic mismatch); student mode declared48 but scored only3 (2 successes,
1 invalid action). The remaining42/45 samples were unscored. Queue entries and
collector finishes are not completed scored windows. The old waiter failed and
the diagnostic unit is absent; neither is an active qualification owner. Do not
rerun its old source/runtime blindly. Retain its exact traces only for the
assigned historical native-return/runtime investigation.

Current obligations are separately assigned: the generation supervisor must
serve the repaired12 TRAIN cases through the existing Qwen guardian/feeder as
explicit group4 authored and group4 student packets, recording actual48 outcomes
per mode plus source, actor, runtime and text-emulation provenance. The student
supervisor must separately qualify actual native positional-KV/hybrid-state
execution at a warranted current-checkpoint boundary. Qwen text emulation cannot
qualify student native state transport. Full48 scored results are still pending;
no qualification or exposure is inferred from registration, queueing or static
reference execution. The production selector rejects `pending-live-48` tasks;
existing admitted train tasks continue throughout this work.

The production RL bootstrap now defaults to the published 128-row train-only
union at `runs/decision-rich-train-20261016-v2-curriculum/`. This scheduling-only
successor retains every original task and adds structurally checked levels
0/1/2/3 with counts 1/3/8/116. Both raw and shared-semantic bootstrap plans use
its exact cases SHA `bfb1b595d013616cfca5c4bc4aab44a218e2398ffc505b1954405dbb56bfd53a`.
Its manifest retains the 72-row
S7 base, 56 admitted broad tasks, and their exact audit and selection proof;
both model entries share the same case SHA. The pending 12-task executor
qualification remains excluded and can be appended through a separately
verified task-source update.

New successful teacher trajectories must come from the substantially stronger
self-hosted CoreWeave Qwen teacher. Student measurement outputs remain diagnostic.
Queue these task inputs for the next teacher window after live qualification;
retain complete verified trajectories, source/split/runtime/actor identities,
and transfer useful successes to both models through the shared RFT mechanism.
Protected held content, IDs and answers were not used to author this packet.

## Validation and limits

### Private writer public-return repair (2026-10-16)

The first 48 self-hosted teacher scaffold executions finished with 44 accepted
trajectories and four failures of `private-writer-return`. Its authored root
declared a private `Neuralese<string>` result but returned it as the task's
public `string`. This is an authored boundary mismatch; those failures do not
measure teacher capability. The private writer now explicitly returns public
`string`, while the writer-reader tasks retain their private state transfer.
The existing host contract now executes this child call as well as the nine
ordinary tasks: all ten pass, and an incorrect ordinary answer is rejected
(Node 24, ledger unit `root-private-writer-public-contract-node24-20261016`).
Task meaning and reward remain natural-language/existing exact
oracles; the repair changes the authored typed reference, not runtime coercion.
The original packet and its four assigned repair traces retain their original
hashes. Rebuilding creates a new packet and contract hash, still pending live
qualification; do not mutate the old packet or relabel its failures as successes.
The repaired qualification packet is 12 rows, SHA
`079b1279232f211294922265e88eca187066899a09360af84e43ace9e9b84dee`;
the repaired pending 84-row union is SHA
`8470aaf16ca346d9a408077dab42397a7e843bed226449d83301b98218eb6d3d`.
Both remain outside the production task source until live qualification.

Focused Python tests exercise growing difficulty, all-solved regression visits,
new hard tasks, pending-source rejection, durable all-solved restart/appending,
and existing controller update/publication recovery. The real natlang host
contract accepts all nine ordinary scaffold oracles plus the repaired private
writer's public return, and rejects an incorrect
answer using a scripted provider. Static checks retain the three typed private
roots and their exact contracts. These checks validate mechanisms and task
oracles; they are not live model performance or completed RL training.

The historical 6750 held report was inspected only for diagnosis: 0/16, with
invalid actions, incomplete/budget and semantic failures, plus known native-ref
write-out debt. Those failures do not establish inability to learn or justify
training on protected evaluation content. Root retains runtime repair and
actual live outcome interpretation.

## Actual teacher/source reconciliation (2026-10-18)

The SHA-bound audit
`/home/werg/data/dgxops/rl-curriculum-current-teacher193-reconciliation-20261018/audit.json`
(SHA-256 `678fc6694503780d066db9cdd73355e850a9e7923398ff32bcf857aa70719ff3`)
authenticates all193 ready entries/11,785 qualified rows. None selects a pending12
case. Registered source-context membership includes all128 admitted tasks and
all12 repaired tasks, but context membership does not prove selection or success.
The repaired12 tasks have zero current held/quality/pending group intersections.

The actual selected-sample join
`/home/werg/data/dgxops/rl-curriculum-current-teacher193-reconciliation-20261018/actual-selected-samples.json`
(SHA-256 `c6714ab853fe0d95239c9e42742ba4e82f6e56572eb3357cd30d9ddf6fc64624`)
binds registered transfer samples to exact executed packet indices, original raw
row/output SHA and unchanged program meaning. It finds **23 accepted prior Qwen
captures** semantically matching the repaired12: two per case except one for
private-two-inputs. All used whole-student-root mode, only sample indices0/1,
and text Neuralese emulation; their original teacher execution identity is
`15f338095d146a8057894e0a8df46376111333250cf63c74f0bc3c7f71326729`.
Keep those useful accepted captures and provenance. They are not the missing
48-sample authored or student windows and do not qualify native transport.
The same audit finds actual selected samples for77 of the128 admitted task IDs
in registered views; this is not a complete-attempt or all-positive claim.
Original negative grades in reviewed source views remain unchanged.
