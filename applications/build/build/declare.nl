---
description: Build graph from declarations. Read the declared tasks of a build as a dependency graph - producers, derived dependencies, sources, dependents - and every fault in the declarations.
args:
  tasks: Task[]
  goal: string
returns: Graph
---
Read tasks, the declared tasks of a build for goal, as a dependency graph. Compute it exactly in eval from these
steps. Every list you produce is sorted by string order and has no duplicates, except problems, which keep the order
they are found in.

1. Ids. ids = the ids of tasks.
   - A task whose id is "" is a problem with code "empty-id".
   - An id that two or more tasks hold is a problem with code "duplicate-id", tasks [that id].
   - A goal that is not in ids is a problem with code "unknown-goal", tasks [goal].
2. Producers. producerOf maps each output path to the ids of the tasks that list it in outputs. A path with more than
   one id is a problem with code "duplicate-output", tasks those ids.
3. For each task t, in the order given:
   - t.argv empty: code "no-command". t.outputs empty: code "no-output".
   - For each id n in t.needs: n equals t.id: code "self-dependency". n not in ids: code "unknown-need". n listed
     twice: code "duplicate-need".
   - A path in both t.inputs and t.outputs: code "self-read".
   - producers(t) = the ids in producerOf[p] for every p in t.inputs, leaving out t.id.
   - Every producer that is not in t.needs: code "undeclared-dependency", tasks [t.id, that producer].
   - sources(t) = the inputs p that no task other than t produces.
   - deps(t) = t.needs (leaving out t.id and unknown ids) together with producers(t).
   Each problem's tasks are the ids involved, and its detail says in one sentence what to change, naming the ids and
   paths: "task goal reads out/a.txt, which task a produces: add a to the needs of goal".
4. dependents(x) = the ids of the tasks t with x in deps(t).
5. nodes = one node per distinct id other than "", sorted by id: { id, needs: t.needs sorted without duplicates and
   without t.id, producers: producers(t), deps: deps(t), dependents: dependents(id), sources: sources(t) }. When an
   id is held by several tasks, use the first.

Return { nodes, problems }.
