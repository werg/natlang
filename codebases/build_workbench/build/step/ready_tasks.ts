/*---
engine: typescript-host
args:
  state: State
returns: Task[]
---*/
const done = new Set(args.state.order);
const byId = new Map(args.state.tasks.map(t => [t.id, t]));
const needed = new Set();
const visit = id => {
  if (needed.has(id)) return;
  needed.add(id);
  for (const parent of byId.get(id)?.needs ?? []) visit(parent);
};
visit(args.state.goal);
return args.state.tasks.filter(t => needed.has(t.id) && !done.has(t.id) &&
  t.needs.every(n => done.has(n)));
