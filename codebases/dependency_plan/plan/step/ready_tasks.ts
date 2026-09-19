/*---
args:
  state: State
returns: Task[]
---*/
const done = new Set(args.state.order);
return args.state.tasks.filter(t => !done.has(t.id) && t.needs.every(n => done.has(n)));
