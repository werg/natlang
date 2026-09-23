export default function prepare(goal: string, tasks: Task[]): State {
const ids = new Set(tasks.map(t => t.id));
const outputs = tasks.flatMap(t => t.outputs);
let detail = '';
if (!ids.has(goal)) detail = `unknown goal: ${goal}`;
else if (ids.size !== tasks.length || tasks.some(t => !t.id)) detail = 'duplicate or empty task ID';
else if (outputs.length !== new Set(outputs).size) detail = 'two tasks declare the same output';
else if (tasks.some(t => !t.argv.length || t.outputs.length === 0)) detail = 'task lacks command or output';
else if (tasks.some(t => t.needs.includes(t.id) || new Set(t.needs).size !== t.needs.length)) detail = 'self or duplicate dependency';
else if (tasks.some(t => t.inputs.some(p => t.outputs.includes(p)))) detail = 'task reads its own output';
return { goal: goal, tasks: tasks, order: [], results: [], blocked: [],
  status: detail ? 'invalid' : 'running', detail };
}
