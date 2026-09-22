/*---
engine: typescript-host
args:
  state: State
returns: boolean
---*/
return args.state.status !== 'running';
