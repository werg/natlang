/*---
args:
  state: State
returns: boolean
---*/
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       Object.keys(args.state.fields).every(key => key.trim().length > 0);
