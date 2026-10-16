# Adaptive whole-execution task curriculum

`selectTrainingTasks(tasks, groupStats, budget, curriculum) -> TaskSelection[]`
is the shared typed natural-language policy for LFM and Mellum.

1. Start with tractable exact-verifier natlang execution. Prefer tasks that can
   yield useful verified successes, then introduce tools, longer executions and
   communication as actual train-task outcomes improve.
2. Reward correct execution regardless of whether communication is used. Task
   difficulty is metadata, never an RL admission edge/node-count requirement.
3. Keep all usable admitted train tasks scheduled: reserve rotating breadth
   work alongside adaptive practice. Revisit solved work after policy changes.
4. Count only scored outcomes when adapting difficulty. Repair infrastructure;
   use verified whole-trajectory teacher/RFT seeding for all-fail groups.
5. Preserve protected held separation. New static task scaffolds remain pending
   model-facing qualification until about48 live executor samples are measured in the actual intended root mode. Authored-root private-call diagnostics do not qualify whole-student-root task prompts.

The current crisp policy adapter implements this interface through existing
`production.select_tasks`, with optional external NL command implementation.
Exact packet/source bindings, row/count validation, deterministic rotation,
selection durability and outcome bookkeeping are crisp mechanisms. The existing
collector, verifier, whole-episode graph capture, shared learner and recipe
remain the only execution/training paths. Static task construction creates
program records with exact host verifiers, not invented student/teacher traces.
