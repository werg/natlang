/*---
engine: typescript-host
args:
  state: State
returns: Bool
---*/
return args.state.status !== 'running';
