---
name: complete-the-call
description: Use when finishing a staged result or choosing success, blocked or failed. Check completion and provide the right result or reason.
---

Use return_result with status "success" only when the task is complete and value has the declared type. An eval may stage a value without finishing: after verifying it, finish by omitting value with status "success", or use eval finish:true for a fresh exact computed result. For missing required information, use status "blocked" with a reason and no value. For contradictory or invalid instructions, use status "failed" with a reason and no value. Repair a rejected stop shape without converting an unfinished plan into success. A valid false, zero or empty result is still success.
