/*---
args:
  state: State
returns: Bool
---*/
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       Object.keys(args.state.fields).every(key => key.trim().length > 0);
