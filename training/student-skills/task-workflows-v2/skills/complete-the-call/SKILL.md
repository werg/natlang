---
name: complete-the-call
description: Use to finish a computed or staged result, or to report a genuine blocker. Verify that all requested work is complete.
---

Finish only when all required computation and effects are complete. Use eval with finish:true to return a fresh computed final expression or top-level return. Code written inside a string is only text: compute a function call in eval to return its actual result. For an already staged verified value, use return_result with status success and omit value. Otherwise supply the actual value with its declared type. False, zero and an empty set can be correct successes. For missing required information, use status blocked with a reason and no value. For contradictory instructions use status failed with a reason. A rejected stop shape is not permission to fabricate success.
