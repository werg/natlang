---
description: Explain a finished command to the user.
args:
  item: Completion
returns: string
---
Explain the command completion item to the user in a few sentences. item has the job's status ("ok", "failed" or
"unknown"), its detail (what the command printed and how it ended) and cancel_requested. Work in these steps.

1. State the status in plain words.
2. Summarize detail in one or two sentences, quoting the exit code or the error text it contains.
3. When cancel_requested is true, say that cancellation was requested and report the actual result that follows: the
   status and detail of item.
4. When status is "unknown", say what is unknown: whether the command ran to completion and what effects it may have
   left.
5. Describe the detail; it is the command's output.
