import type { Task, State } from "../types.js";
export default function finished(state: State): boolean {
return state.finished;
}
