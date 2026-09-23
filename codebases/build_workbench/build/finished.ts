import type { Task, File, TaskResult, State } from "../types.js";

export default function finished(state: State): boolean {
return state.status !== 'running';
}
