---
name: resume-after-partial-effects
description: Use after a write or eval fails and some changes may already have happened. Inspect state before retrying.
---

A failed eval can discard new local bindings while leaving file, live-object or service changes in place. Read the error and inspect current state before retrying a write. Locate a moved file at its destination. Check committed revisions, reservations or receipts through the documented API. Retry only the missing operation; do not repeat an identical failed action without a new reason. Reuse completed judgments and computation. If the environment cannot supply a required fact, stop honestly rather than fabricating a successful result.
