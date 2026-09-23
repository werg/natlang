export default async function advance(_state: State, chosen: string): Promise<State> {
const state = _state;
const done = new Set(state.order);
const task = state.tasks.find(t => t.id === chosen);
if (state.status !== 'running' || !task || done.has(task.id) || !task.needs.every(n => done.has(n)))
  return { ...state, status: 'invalid', detail: `chosen task is not ready: ${chosen}` };
const result = await host.build.execute(task);
const results = [...state.results, result];
if (result.status !== 'ok')
  return { ...state, results, status: result.status === 'unknown' ? 'unknown' : 'failed',
    detail: `${task.id}: ${result.detail}` };
const order = [...state.order, task.id];
return { ...state, order, results, status: task.id === state.goal ? 'done' : 'running', detail: '' };
}
