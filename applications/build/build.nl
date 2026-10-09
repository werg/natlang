---
description: A build system in natural language. Build goal from the declared tasks - read the dependency graph, plan the goal's closure and order, run ready tasks one round at a time with cache reasoning and failure diagnosis, and report.
args:
  goal: string
  tasks: Task[]
  files?: Folder
returns: BuildReport
---
Build goal from tasks with the stages in your folder. The build service is the workspace that runs commands and keeps
the ledger of earlier runs; the stages decide. files, when given, is the folder the scheduling policy may read a
task's named local input from. Work in eval and return the report as an object built from the values below.

1. Graph. graph = declare(tasks, goal). Its problems are the faults in the declarations. When there are any, run
   nothing and return the report with status "invalid", detail = the problems' details joined with "; ", problems =
   graph.problems.
2. Plan. needed = closure(graph, goal), then plan = order(graph, goal, needed). When plan.cycle is not empty, run
   nothing and return the report with status "blocked", blocked = needed, detail = "tasks wait on each other: " and
   plan.cycle joined with ", ", plan.
3. State. state = { goal, tasks, graph, needed, order: [], attempted: [], results: [], judgments: [], blocked: [],
   status: "running", detail: "" }.
4. Rounds. final = await step.iterateOn(state, files)
   .withMeasure(s => 2 * (s.needed.length - s.attempted.length) + (s.status === "running" ? 1 : 0))
   .until(s => s.status !== "running").
   Each round attempts one task or stops the build, so the measure falls every round and the loop ends.
5. Report. closing = summarize(final). Return { goal, status: final.status, detail: final.detail, order: final.order,
   results: final.results, blocked: final.blocked, built: the tasks of final.judgments whose action is "ran", in
   order, reused: those whose action is "reused", not_run: the ids of final.needed that are not in final.attempted,
   judgments: final.judgments, diagnosis: final.diagnosis (leave the field out when there is none), problems: [],
   plan, summary: closing.summary, next: closing.next }.
   For an invalid or blocked build the same fields hold: empty lists for order, results, built, reused and
   judgments, not_run = needed (or [] when the graph is invalid), and summary and next from summarize applied to the
   state { goal, tasks, graph, needed, order: [], attempted: [], results: [], judgments: [], blocked, status, detail }.
