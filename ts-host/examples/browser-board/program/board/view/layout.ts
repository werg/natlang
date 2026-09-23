import type { Task, Board, UiEvent, Decision, TaskGroup, ViewPlan, UiAction, UiNode } from "../types.js";

export default function layout(state: Board, plan: ViewPlan): UiNode {
const byId = new Map(state.items.map(item => [item.id, item]));
const listed = plan.groups.flatMap(group => group.ids);
if (!plan.title || !plan.summary || !Array.isArray(plan.groups) ||
    new Set(listed).size !== state.items.length ||
    listed.some(id => !byId.has(id)) || state.items.some(item => !listed.includes(item.id)))
  throw new Error('view plan does not cover the task board');
const children = [
  { tag: 'h1', text: plan.title },
  { tag: 'p', text: plan.summary },
  { tag: 'input', id: 'command', label: 'Command',
    placeholder: 'Add a task, or describe a change' },
  { tag: 'button', text: 'Apply command', action: { kind: 'command', from: 'command' } },
  { tag: 'button', text: 'Clear completed', action: { kind: 'clear_done' } },
];
for (const group of plan.groups) {
  children.push({ tag: 'section', children: [
    { tag: 'h2', text: group.label },
    { tag: 'ul', children: group.ids.map(id => {
      const item = byId.get(id);
      return { tag: 'li', children: [
        { tag: 'span', text: item.text + (item.done ? ' (done)' : '') },
        { tag: 'button', text: item.done ? 'Reopen' : 'Complete',
          action: { kind: 'toggle', value: item.id } },
      ] };
    }) },
  ] });
}
return { tag: 'main', children };
}
