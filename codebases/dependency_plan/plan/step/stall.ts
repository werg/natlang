import type { Task, State } from "../../types.js";
export default function stall(state: State): State {
return {...state, blocked: state.tasks.filter(t => !state.order.includes(t.id)).map(t => t.id), finished: true};
}
