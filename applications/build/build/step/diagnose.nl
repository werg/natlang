---
description: Build failure triage. Say why a task failed or was interrupted, from the workspace's result and the task's declaration, and what to change.
args:
  task: Task
  result: TaskResult
  graph: Graph
returns: CheckedDiagnosis
---
Diagnose result, the outcome of task that did not finish ok. result.detail is the workspace's message: for a command
it is "exit N: " and the command's output, or the process error; for a refusal it names the rule and the path.
Work from result.detail, task (argv, inputs, outputs) and graph (the task's node and its dependents).

1. cause. Take the first that matches:
   - result.status is "unknown": "interrupted".
   - detail starts "output already exists": "output-conflict"; culprit the path.
   - detail starts "declared input changed" or "input changed during": "input-mutated"; culprit the path.
   - detail has "escapes workspace" or "enters host cache": "path-escape"; culprit the path.
   - detail starts "invalid task declaration" or "invalid built-in": "declaration-error".
   - detail has "ENOENT" or "not found" for the executable (the error names task.argv[0]): "missing-tool"; culprit
     task.argv[0].
   - detail has "ENOENT" or "no such file" or "is not a file" for a path in task.inputs: "missing-input"; culprit
     that path.
   - result.exit_code is not 0 and not -1: "command-failed"; culprit the file or tool named in the command's
     output, else "".
   - anything else: "other".
2. summary: one sentence a person can act on, naming the task, the cause and the culprit.
3. fix: one sentence naming what to change: for an output-conflict, remove the file or declare the task's output
   afresh; for input-mutated, make the command leave its inputs alone or declare the file as an output; for
   missing-input, the task that should produce the path (the graph's producers say which) or the file to supply; for
   missing-tool, the tool to install or the path to correct; for command-failed, what the output says is wrong;
   for interrupted, how to find out what the command did.
4. retry: "inspect-first" when status is "unknown" (the command may have changed files; check task.outputs before
   running it again). "after-fix" for a cause that a change removes. "no" when running again changes nothing and
   nothing can be changed (other).

Return { task: task.id, cause, culprit, summary, fix, retry }.
