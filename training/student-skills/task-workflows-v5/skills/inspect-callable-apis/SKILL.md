---
name: inspect-callable-apis
description: Use for an unfamiliar callable API or an unknown method or argument error. Inspect its declaration, then make a documented call.
---

Read the declaration of the specific listed function or service with read_code. Then use it: reading documentation alone does not complete the task. Use the actual variables in scope. A declaration such as f(item: Item): number describes the call f(item); do not paste parameter names or return-type annotations into a call expression. Do not invent methods such as state or status. After an error, change your next action using the error and the declaration. Read the same unchanged declaration again only if you need information no longer visible. Use eval to execute calls and inspect their actual results.
