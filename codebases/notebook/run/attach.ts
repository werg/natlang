/*---
engine: typescript-host
args:
  state: NotebookState
  answer: string
returns: NotebookState
---*/
return { ...args.state, answer: args.answer };
