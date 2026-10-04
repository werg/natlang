---
name: inspect-callable-apis
description: Use when calling an unfamiliar function, service or built-in. Read its declaration before choosing methods and arguments.
---

Use read_code with the listed function, module or built-in name to inspect its declaration. Use the variables already declared in your eval scope. Call methods with the documented arguments; do not infer properties such as state, status or result from their names. Check whether a method expects an item, destination or quantity. After a TypeError or unknown-member error, inspect the declaration and current bindings before trying again. Prefer one documented call over speculative exploration.
