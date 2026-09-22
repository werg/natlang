/*---
engine: typescript-host
args:
  state: ConsoleState
  request: string
  result: NotebookState
returns: ConsoleState
---*/
return { requests: [...args.state.requests, args.request],
  runs: [...args.state.runs, args.result], status: args.result.status };
