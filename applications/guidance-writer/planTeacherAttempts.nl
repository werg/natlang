---
description: Plan how many privileged teacher attempts each scored student group gets, within a total budget.
args:
  groups: TeacherGroup[]
  budget: number
returns: TeacherAttempts[]
---
1. Each group in groups is one training world with its students' outcomes: students attempts, successes of them, meanQuality between 0 and 1, the runtime feedback the attempts received, and facts, the number of verified facts the world has.
2. A teacher attempt runs the same world with guidance written from those facts, so it is useful where the students mostly failed and facts is above 0. A group whose students all succeeded gains little; a group with no facts gets no attempts.
3. Order the useful groups from most to least informative: fewer successes and lower meanQuality first, and among those, groups whose feedback shows a concrete mistake before groups without feedback.
4. Give each group in that order 0, 1 or 2 attempts, 2 for the most informative failures, until the attempts given add up to budget. Never exceed budget in total.
5. Return {group, attempts} objects using the group numbers from groups, each group at most once.
