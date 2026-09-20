/*---
engine: typescript-host
args:
  request: PackageRequest
  locks: Lock[]
  selected: Text
returns: Resolution
---*/
const lock = args.locks.find(row => row.id === args.selected);
if (!lock) return { status: 'invalid-selection',
  lock: { id: '', root: args.request.name, engine: args.request.engine, packages: [] },
  alternatives: args.locks.length, detail: 'selected ID was not offered by the exact solver' };
return { status: 'resolved', lock, alternatives: args.locks.length, detail: '' };
