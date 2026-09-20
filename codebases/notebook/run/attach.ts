/*---
engine: typescript-host
args:
  state: NotebookState
  answer: Text
returns: NotebookState
---*/
return { ...args.state, answer: args.answer };
