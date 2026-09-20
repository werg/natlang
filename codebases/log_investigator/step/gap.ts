/*---
engine: typescript-host
args:
  acc: IncidentState
  item: LogEvent
returns: IncidentState
---*/
if (args.item.cursor <= args.acc.cursor) return args.acc;
return { ...args.acc, cursor: args.item.cursor, status: 'gap',
  unknowns: [...args.acc.unknowns, `Source gap at cursor ${args.item.cursor}: ${args.item.message}`] };
