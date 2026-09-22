/*---
args:
  state: State
returns: boolean
---*/
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       args.state.members.every(member => member.trim().length > 0) &&
       new Set(args.state.members).size === args.state.members.length;
