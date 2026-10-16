---
description: Select train tasks for adaptive policy practice and complete rotating breadth.
args:
  tasks: TrainingTask[]
  groupStats: TrainingGroupOutcome[]
  budget: TrainingSelectionBudget
  curriculum: TrainingCurriculumOptions
returns: TrainingTaskSelection[]
---
1. Use only the named train tasks. Read scored outcomes as task-choice telemetry; unscored outcomes say nothing about the model's skill.
2. Begin practice at the lowest task level without observed success. Move practice to harder levels when the scored success rate reaches curriculum.promotion_success_rate. Prefer mixed-success tasks with useful learning signal; use verified whole-trajectory seeding when failures have no successful samples.
3. Reserve curriculum.breadth_fraction of the selected rows, rounded up, for consecutive tasks beginning at budget.offset and wrapping around tasks. This rotating breadth includes solved tasks and all harder tasks; never permanently retire a task.
4. Fill the remaining rows with distinct useful practice tasks near the current level. Keep the same budget.group samples for every task, at least two. Select no more than budget.rows tasks.
5. Put the rotating breadth rows last, ending with its final consecutive row so the next breadth cursor advances. Return only objects {row, samples}.
6. Correct task execution earns reward whether Neuralese is used or not. When communication occurs, the existing learner covers the entire actual episode graph. Task choice never authorizes dropping a producer or selecting only a consumer.
