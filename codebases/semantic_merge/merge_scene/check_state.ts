/*---
args:
  state: State
returns: Bool
---*/
const ids = args.state.objects.map(object => object.id);
return Number.isSafeInteger(args.state.revision) && args.state.revision >= 0 &&
       ids.every(id => id.trim().length > 0) && new Set(ids).size === ids.length &&
       args.state.objects.every(object => Number.isSafeInteger(object.x) && Number.isSafeInteger(object.y));
