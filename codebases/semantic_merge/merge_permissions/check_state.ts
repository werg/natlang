/*---
args:
  state: State
returns: Bool
---*/
const keys = args.state.rules.map(r => JSON.stringify([r.subject, r.resource, r.action]));
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       args.state.rules.every(r => r.subject.trim() && r.resource.trim() && r.action.trim()) &&
       new Set(keys).size === keys.length;
