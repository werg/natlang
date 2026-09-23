export default function ready_tasks(state: State): Task[] {
const done = new Set(state.order);
const byId = new Map(state.tasks.map(t => [t.id, t]));
const needed = new Set();
const visit = id => {
  if (needed.has(id)) return;
  needed.add(id);
  for (const parent of byId.get(id)?.needs ?? []) visit(parent);
};
visit(state.goal);
return state.tasks.filter(t => needed.has(t.id) && !done.has(t.id) &&
  t.needs.every(n => done.has(n)));
}
