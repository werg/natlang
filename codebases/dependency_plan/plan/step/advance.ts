export default function advance(state: State, chosen: string): State {
const task = state.tasks.find(t => t.id === chosen);
if (!task || state.order.includes(task.id) || !task.needs.every(n => state.order.includes(n)))
  throw new Error("chosen task is not ready");
const order = [...state.order, task.id];
return {...state, order, finished: order.length === state.tasks.length};
}
