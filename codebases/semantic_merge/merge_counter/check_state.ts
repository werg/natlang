/*---
args:
  state: State
returns: Bool
---*/
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       Number.isSafeInteger(args.state.value) && args.state.unit.trim().length > 0;
