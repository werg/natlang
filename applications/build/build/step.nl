---
description: One round of a build. Decide from the state which ready task goes next and whether its recorded outputs are still valid, run or reuse it, diagnose a failure, and commit the next state.
args:
  state: BuildState
  files?: Folder
returns: BuildState
---
Run one round of the build in state and return the next state. Decide from state as it is, perform one effect, and
commit. Work in eval. build is the workspace service. files, when given, is the folder choose may read a task's named
input from. The stages ready, choose, validity and diagnose judge. commit holds the exact guard and the bookkeeping.

1. Ready. ids = await ready(state.graph, state.needed, state.order).
   When ids is empty, nothing can run: return commit.settle(state, { kind: "stop", task: "", why: "no declared task
   is ready" }, null, null).
2. Choose. candidates = the tasks of state.tasks whose id is in ids. picked = await choose(candidates, state.goal,
   files). task = the candidate whose id is picked.
3. Admit. admission = commit.admit(state, { kind: "run", task: picked, why: "" }). When admission.ok is false, return
   commit.settle(state, { kind: "reject", task: picked, why: admission.problem }, null, null).
4. Judge. evidence = await build.inspect(task). verdict = await validity(task, evidence). decision = { kind: "reuse"
   when verdict.valid, else "run", task: picked, why: verdict.reason }. A task that ran makes the outputs its
   dependents read new, so each dependent is judged from the digests it finds when its own turn comes.
5. Effect. A reuse: result = await build.reuse(task). When result.status is not "ok", the ledger no longer matches:
   set decision to { kind: "run", task: picked, why: result.detail } and run it as below. A run: result = await
   build.execute(task).
6. Diagnose. When result.status is not "ok", diagnosis = await diagnose(task, result, state.graph). Otherwise
   diagnosis = null.
7. Commit. Return commit.settle(state, decision, result, diagnosis). Its value is the next state.
