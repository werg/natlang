/*---
engine: typescript-host
args:
  state: State
returns: State
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
const blocked = [...needed].filter(id => !done.has(id)).sort();
return { ...args.state, blocked, status: 'blocked',
  detail: 'No declared task is ready; dependencies may be missing or cyclic.' };
