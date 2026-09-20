/*---
engine: typescript-host
args:
  goal: Text
  tasks: Task[]
returns: State
---*/
const ids = new Set(args.tasks.map(t => t.id));
const outputs = args.tasks.flatMap(t => t.outputs);
let detail = '';
if (!ids.has(args.goal)) detail = `unknown goal: ${args.goal}`;
else if (ids.size !== args.tasks.length || args.tasks.some(t => !t.id)) detail = 'duplicate or empty task ID';
else if (outputs.length !== new Set(outputs).size) detail = 'two tasks declare the same output';
else if (args.tasks.some(t => !t.argv.length || t.outputs.length === 0)) detail = 'task lacks command or output';
else if (args.tasks.some(t => t.needs.includes(t.id) || new Set(t.needs).size !== t.needs.length)) detail = 'self or duplicate dependency';
else if (args.tasks.some(t => t.inputs.some(p => t.outputs.includes(p)))) detail = 'task reads its own output';
return { goal: args.goal, tasks: args.tasks, order: [], results: [], blocked: [],
  status: detail ? 'invalid' : 'running', detail };
