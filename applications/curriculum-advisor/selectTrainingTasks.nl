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
2. Maintain a separate practice level for each declared semantic family (curriculum.frontier_scope is semantic-family). Never hold a mastered family at an easier level because an unrelated family is struggling. Missing family metadata belongs to the one unspecified bucket. Prefer mixed-success tasks with useful learning signal; use verified whole-trajectory seeding when failures have no successful samples.
   - When curriculum.schema is natlang.rl-task-curriculum/1: within a family, begin at its lowest task level without observed success and move to harder levels when that family's scored success rate reaches curriculum.promotion_success_rate.
   - When curriculum.schema is natlang.rl-task-curriculum/2:
     a. A family with no scored task starts at the median of its task levels (curriculum.new_family_start is median), or its lowest level when it is easiest.
     b. A family with scored tasks practices its lowest level whose scored success rate is below curriculum.promotion_success_rate; when there is none, the first level above its highest promoted level, or its top level. Easier levels skipped by the median start count as passed.
     c. Group the scored outcomes by observed_round and walk back from the newest round. Each consecutive round in which fewer than curriculum.spread_target of the groups have spread moves every family's practice level one step: up when most of its groups without spread succeeded, down when most failed. Stop at a round with enough spread, at a change of direction, or after curriculum.difficulty_shift_limit steps. Keep each family's level within its own task levels.
3. Reserve curriculum.breadth_fraction of the selected rows, rounded up, for consecutive tasks beginning at budget.offset and wrapping around tasks. This rotating breadth includes solved tasks and all harder tasks; never permanently retire a task.
4. Of the remaining practice slots, reserve curriculum.family_exploration_fraction, rounded up, for unvisited tasks from least-covered declared families. Count coverage as distinct current-source task IDs with scored outcomes, not sample counts or historical IDs absent from tasks. For each family with an unvisited task outside rotating breadth, choose one representative: under /1 its easiest admitted level when the family has no scored tasks, otherwise (and always under /2) nearest that family's practice level. Prefer families with fewer scored task IDs; break ties under /1 by easiest level for unseen families, otherwise distance to that family's practice level, then the practice priority and cyclic distance from budget.offset. A family with no unvisited tasks gets no exploration slot. The family "unspecified" denotes one unknown bucket, not a separate invented family for each task.
5. Fill other practice slots near each task family's current level, preferring mixed-success tasks, then unvisited tasks, then all-fail tasks, then solved tasks. Prefer a family not already selected for practice; only repeat practice families when distinct ones cannot fill the budget. Keep the same budget.group samples for every task, at least two. Select distinct rows, no more than budget.rows or tasks.length, and do not duplicate rotating breadth rows in practice.
6. Put the rotating breadth rows last, ending with its final consecutive row so the next breadth cursor advances. Return only objects {row, samples}.
7. Correct task execution earns reward whether Neuralese is used or not. When communication occurs, the existing learner covers the entire actual episode graph. Task choice never authorizes dropping a producer or selecting only a consumer.
