import type { Task, File, TaskResult, State } from "../../types.js";

export default function stall(state: State): State {
const done = new Set(state.order);
const byId = new Map(state.tasks.map(t => [t.id, t]));
const needed = new Set();
const visit = id => {
  if (needed.has(id)) return;
  needed.add(id);
  for (const parent of byId.get(id)?.needs ?? []) visit(parent);
};
visit(state.goal);
const blocked = [...needed].filter(id => !done.has(id)).sort();
return { ...state, blocked, status: 'blocked',
  detail: 'No declared task is ready; dependencies may be missing or cyclic.' };
}
