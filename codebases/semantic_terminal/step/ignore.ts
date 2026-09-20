/*---
engine: typescript-host
args:
  acc: Session
  item: Event
returns: Session
---*/
return { ...args.acc, messages: [...args.acc.messages, `Ignored unknown event kind: ${args.item.kind}`] };
