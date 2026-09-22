/*---
engine: typescript-host
args:
  state: NotebookState
returns: boolean
---*/
return args.state.status !== 'running';
