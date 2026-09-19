/*---
args:
  tasks: Task[]
returns: State
---*/
const ids = args.tasks.map(t => t.id);
if (new Set(ids).size !== ids.length) throw new Error("duplicate task IDs");
return {tasks: args.tasks, order: [], blocked: [], finished: args.tasks.length === 0};
