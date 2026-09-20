/*---
engine: typescript-host
args:
  state: State
  chosen: Text
returns: State
---*/
const state = args.state;
const done = new Set(state.order);
const task = state.tasks.find(t => t.id === args.chosen);
if (state.status !== 'running' || !task || done.has(task.id) || !task.needs.every(n => done.has(n)))
  return { ...state, status: 'invalid', detail: `chosen task is not ready: ${args.chosen}` };
const result = await host.build.execute(task);
const results = [...state.results, result];
if (result.status !== 'ok')
  return { ...state, results, status: result.status === 'unknown' ? 'unknown' : 'failed',
    detail: `${task.id}: ${result.detail}` };
const order = [...state.order, task.id];
return { ...state, order, results, status: task.id === state.goal ? 'done' : 'running', detail: '' };
