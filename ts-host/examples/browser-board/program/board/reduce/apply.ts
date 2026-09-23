export default function apply(_state: Board, event: UiEvent, _decision: Decision): Board {
const state = _state;
const decision = _decision;
const items = state.items.map(item => ({ ...item }));
if (decision.kind === 'add') {
  const text = String(decision.text || '').trim();
  if (!text || text.length > 500) throw new Error('task text must have 1–500 characters');
  items.push({ id: `task-${state.next_id}`, text, done: false });
  return { ...state, items, next_id: state.next_id + 1, revision: state.revision + 1 };
}
if (decision.kind === 'toggle' || decision.kind === 'remove') {
  const index = items.findIndex(item => item.id === decision.item_id);
  if (index < 0) throw new Error('unknown task');
  if (decision.kind === 'toggle') items[index].done = !items[index].done;
  else items.splice(index, 1);
  return { ...state, items, revision: state.revision + 1 };
}
if (decision.kind === 'clear_done')
  return { ...state, items: items.filter(item => !item.done), revision: state.revision + 1 };
if (decision.kind === 'ignore') return state;
throw new Error('unknown decision');
}
