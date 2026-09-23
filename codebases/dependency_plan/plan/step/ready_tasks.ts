import type { Task, State } from "../../types.js";
export default function ready_tasks(state: State): Task[] {
const done = new Set(state.order);
return state.tasks.filter(t => !done.has(t.id) && t.needs.every(n => done.has(n)));
}
