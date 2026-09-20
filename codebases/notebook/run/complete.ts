/*---
engine: typescript-host
args:
  state: NotebookState
returns: Bool
---*/
return args.state.status !== 'running';
