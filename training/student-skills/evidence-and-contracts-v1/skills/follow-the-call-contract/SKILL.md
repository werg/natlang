---
name: follow-the-call-contract
description: Use when selecting an unfamiliar callable, repairing an argument error, or returning a computed value with a required type.
---

Inspect the declaration of the specific callable you need. Invoke it through eval with actual values; do not paste its declaration or invent methods. After an error, change the next action using that error. Complete all required effects before returning. A finish:true eval ends the call with its fresh computed result. Otherwise use a plain typed answer or an actual return_result tool call. Printed invocation text does not call a tool. False, zero and empty collections may be correct successes.
