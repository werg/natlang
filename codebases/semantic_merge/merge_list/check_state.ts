/*---
args:
  state: State
returns: boolean
---*/
const ids = args.state.items.map(item => item.id);
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && new Set(ids).size === ids.length;
