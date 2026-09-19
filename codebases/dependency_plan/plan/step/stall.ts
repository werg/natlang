/*---
args:
  state: State
returns: State
---*/
return {...args.state, blocked: args.state.tasks.filter(t => !args.state.order.includes(t.id)).map(t => t.id), finished: true};
