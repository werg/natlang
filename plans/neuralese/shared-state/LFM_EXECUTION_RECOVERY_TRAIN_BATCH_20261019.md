# LFM execution-recovery TRAIN batch (2026-10-19)

## Natlang-native design

This batch adds independent TRAIN program inputs, not teacher outputs. Each program is a natural-language task over a small, newly constructed service world. The source builder supplies a complete executable reference and exact result oracle; the existing Qwen teacher queue samples the responses later. No held input, answer, response or trace enters construction.

The ordinary track uses the existing `natlang.program/2` and `teacher_source_sequence` contracts. Its root reads both pages, asks the typed `select_record` function about every row, retains the selected IDs and values, writes one typed summary, reads it back, and returns it. Each world is small enough for a short complete trajectory, spans all eight existing task domains, and includes independent seeded runtime feedback variants for a bad page range, a reserved service binding, and a mismatched typed result. Exact JavaScript service boundaries and the Python source/oracle verifier are crisp because they describe outside-world behavior and exact grading; task choice and recovery remain natural-language policy.

The communication track uses the existing authored-root trajectory contract and `nl.with` typed private state. A sampled producer writes an evidence string; a sampled consumer reads that actual state and returns the exact requested public value. The reference requires the full producer/consumer call sequence. These are candidate graph tasks; only captured teacher trajectories with actual writer-to-reader state edges can count as complete graphs. They are not synthetic graph claims or a new transport implementation.

## Existing mechanism mapping

- Fresh world facts and complete references: `scripts/neuralese_data/build_fresh_recovery_teacher_worlds.py` plus one sibling profile builder.
- Exact reference qualification and protection closure: existing current-protection teacher reference qualifier.
- Actual sampled responses: existing Qwen source sequence and guardian, owned by the generation queue owner.
- Per-row quality/admission and split decisions: existing teacher trajectory review and `teacher_source_sequence`; reference qualification alone never admits training.
- Training: existing shared RL/RFT and native whole-trajectory mechanisms; no new optimizer, reward, queue, server or scheduler.

## Validation and release boundary

Tests must verify deterministic distinct IDs, all-train rows, current held/quality/pending exclusion, short page coverage, exact oracle equality, seeded failure recovery requirements, and graph contracts with producer/consumer edges. Run the existing exact authored-reference qualification on the generated packet before handing off. Teacher execution, teacher acceptance, training admission and actual student exposure remain separate milestones.

## Short verify-and-return lane (2026-10-20 decomposition)

`finishVerifiedWork(input, folder) -> number` is the additional typed natural-language
function. Four independently seeded TRAIN worlds each offer three tasks: read an
already correct numeric record and return it; write the supplied value, read it
back, and return it; recover from seeded typed-result feedback and return the
actual numeric service fact. These require no child or communication edge. Correct
task execution is sufficient, and any actual graph still uses whole-episode RL.

The existing builder constructs exact facts and references, the existing reference
executor checks them, and the existing self-hosted Qwen queue supplies all sampled
teacher responses. JSON/oracle/protection verification stays crisp. This adds no
new reward, termination heuristic, queue, trainer, or runtime instruction. Select
this source profile explicitly; existing 24-task and private-state packets retain
their identities. Twelve source tasks with four independent samples provide the
48-execution live measurement window. New model-facing task text is candidate
debt until the exact source/mode/runtime measurement and teacher admission; source
construction and reference success do not qualify training or student lift.

The final 12-task source passed the existing actual reference executor on
2026-10-20 (12/12 accepted, zero model calls); the typed-feedback cases use an
actual compiler-rejected assignment before successful recovery. A bare string
return was rejected during preparation because it did not provide that recovery
contract. Current candidate and reference custody live under
`/home/werg/data/dgxops/verify-return-curriculum-candidate-v3-20261020/verify-return/`.
The 48 strong-teacher requests may proceed in the next own-Qwen generation
window, qualifying the actual teacher/runtime and verified trajectories.
Student RL separately measures the actual trained actor in useful whole-root
rollouts; neither measurement waits for unrelated communication readiness.
Both remain pending; no source or trajectory is admitted by this change.

## Computed typed results and successful-feedback completion (2026-10-20)

Design first: `computeRequestedResult(facts, folder) -> Result` is a typed
natural-language task, with number, boolean, string and record result worlds.
Each independently seeded TRAIN world has three modes: compute from service
facts; verify a previously saved result and return the typed value; recover
from one deliberately invalid typed assignment, compute, save and verify the
result. The function chooses its implementation and may use ordinary execution
or Neuralese. Correct execution is rewarded either way; real graph episodes
retain every producer, consumer and outcome for the existing whole-graph update.

The existing source constructor supplies exact service facts, typed signatures,
JSON files, references and deterministic oracles. Existing protection joins,
reference executor, generation guardian, source sequence and shared trainers
remain the mechanisms. No new queue, reward, runtime prompt or termination
budget is introduced. All sampled trajectories must come from the existing
self-hosted stronger Qwen teacher. Twelve candidate tasks times four samples
constitute the live model-facing measurement; reference replay is source
validation only. A new source profile preserves previous packet identities and
all existing task scheduling. Held data supplies no task text, facts or answers.
