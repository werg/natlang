/*---
args:
  state: State
  chosen: string
returns: State
---*/
const task = args.state.tasks.find(t => t.id === args.chosen);
if (!task || args.state.order.includes(task.id) || !task.needs.every(n => args.state.order.includes(n)))
  throw new Error("chosen task is not ready");
const order = [...args.state.order, task.id];
return {...args.state, order, finished: order.length === args.state.tasks.length};
