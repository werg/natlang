---
args: { isolation: IsolationCheckpoint, outgoing: string, incoming: string }
returns: IsolationCheckpoint
---
Carry the precise isolation state and lock references to the incoming shift. Name the required check owner and preserve each handover check; shift change alone never authorizes removing isolation.
