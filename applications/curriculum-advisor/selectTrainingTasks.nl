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
4. Of the remaining practice slots, reserve curriculum.family_exploration_fraction, rounded up, for unvisited tasks from least-covered declared families. Count coverage as distinct current-source task IDs with scored outcomes, not sample counts or historical IDs absent from tasks. For each family with an unvisited task outside rotating breadth, choose one representative: its easiest admitted level when the family has no scored tasks, otherwise nearest the practice level. Prefer families with fewer scored task IDs; break ties by easiest level for unseen families, otherwise distance to the practice level, then the practice priority and cyclic distance from budget.offset. A family with no unvisited tasks gets no exploration slot. The family "unspecified" denotes one unknown bucket, not a separate invented family for each task.
5. Fill other practice slots near the current level, preferring mixed-success tasks, then unvisited tasks, then all-fail tasks, then solved tasks. Prefer a family not already selected for practice; only repeat practice families when distinct ones cannot fill the budget. Keep the same budget.group samples for every task, at least two. Select distinct rows, no more than budget.rows or tasks.length, and do not duplicate rotating breadth rows in practice.
6. Put the rotating breadth rows last, ending with its final consecutive row so the next breadth cursor advances. Return only objects {row, samples}.
7. Correct task execution earns reward whether Neuralese is used or not. When communication occurs, the existing learner covers the entire actual episode graph. Task choice never authorizes dropping a producer or selecting only a consumer.
