---
description: Schedule the day for a request, the way a planner does. Read what the request requires and wishes, work out where each task can go, build and check complete schedules, repair near misses, rank them by the wishes and explain the choice.
args:
  request: string
  view: DayView
  settings: Settings
returns: CheckedProposal
---
Plan the day in view for request, with the stages in your folder; calendar checks what they produce. view is the day
at one revision and all times are minutes after view.origin. settings.candidates is how many complete schedules to
build and compare at most. The result is a Proposal. The host commits it only if calendar accepts it, so a plan
you did not check with calendar.check is not returned.

Reading the request:
1. tasks = readTasks(request, view). all = view.tasks followed by tasks.tasks. Then, at the same time,
   limits = readLimits(request, view, all) and wishes = readPreferences(request, all).
2. hard = { tasks: tasks.tasks, limits: limits.limits, blocks: limits.blocks }. questions = the questions of tasks,
   limits and wishes, joined in that order. When questions is not empty, return status "unclear" with placements [],
   hard, the questions, ranking [], considered 0, truncated false, and explanation "To plan this I need to know: "
   followed by the questions.

Constraints:
3. domains = domains(view, hard). ordering = order(domains). When ordering.problem is not empty, return status
   "unclear" with that problem followed by "; which dependency should change?" as the only question (and as the
   explanation), the other fields as in step 2.

Candidates:
4. offered = enumerate(domains, ordering.order, view.slot, hard, settings.candidates). Its options are candidate
   schedules: { id, placements }. truncated = offered.truncated.
5. Check every option at once: verdict = calendar.check(option.placements, hard). valid holds the options whose verdict
   is ok. The others are broken, each with its violations.
6. When valid is empty and broken is not, repair the broken option with the fewest violations. Repeat at most 3 rounds
   (a counted loop): placements = repair(placements, violations, domains, ordering.order, view.slot), then
   verdict = calendar.check(placements, hard); stop at the first round whose verdict is ok, and then valid is that one
   option with the repaired placements. Each later round repairs the latest placements with the latest violations.
7. When valid is still empty, ask for the exact search once: confirmed = calendar.exact(hard, settings.candidates). Its
   options satisfy every constraint: valid = confirmed.options and truncated = confirmed.truncated.
8. When valid is still empty, no schedule exists. diagnosis = diagnose(request, view, hard, domains). Return status
   "infeasible" with placements [], hard, explanation diagnosis.conflict followed by the relaxations as one sentence,
   questions [], ranking [], considered 0, truncated false.

Ranking:
9. Each valid option's description is calendar.describe(option.placements).
10. For every pair of a valid option and a wish in wishes.preferences, at the same time: fit =
    assess(description, wish). That is an Assessment { candidate: option.id, preference: wish.id, fit }.
11. An option's total is the sum over its assessments of wish.weight times 2 for met, times 1 for partly, times 0 for
    missed. With no wishes every total is 0. ranking is { candidate, total } for every valid option, highest total
    first, ties in the order of valid.
12. best = the options whose total equals the highest. With several, keep the first, and for each other in order:
    when tiebreak(request, description of the kept one, description of the other) is "second", the other becomes the
    kept one. The kept one is chosen.

Answering:
13. explanation = explain(request, the chosen description, wishes.preferences, the chosen option's assessments,
    valid.length, truncated).
14. Return status "chosen" with the chosen option's placements, hard, explanation, questions [], ranking,
    considered valid.length and truncated.
