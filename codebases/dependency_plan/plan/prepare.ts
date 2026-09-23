import type { Task, State } from "../types.js";
export default function prepare(tasks: Task[]): State {
const ids = tasks.map(t => t.id);
if (new Set(ids).size !== ids.length) throw new Error("duplicate task IDs");
return {tasks: tasks, order: [], blocked: [], finished: tasks.length === 0};
}
