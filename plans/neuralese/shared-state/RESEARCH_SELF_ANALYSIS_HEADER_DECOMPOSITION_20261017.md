# Preserve self-addressed research analysis

The acquired research source contains assistant messages whose recipient is
literally `assistant<|channel|>analysis`, with another value in its channel field.
That recipient addresses the assistant and embeds its analysis channel; it does
not identify an external tool dispatch. Treating it as a tool adds an invented
pending call and rejects a later genuine observation.

1. Recognize only that exact assistant recipient, without explicit tool calls.
2. Keep the original content as analysis, with no tool call and no pending
   observation. Keep every message and its position.
3. Record original recipient/channel, normalized fields and exact text SHA in
   the existing acquired research lineage beside all original channel metadata.
4. Keep genuine dispatch/observation identity checks strict. Do not infer missing
   calls, repair arguments, reorder observations, or alter other headers.
5. Recover only independently verified omitted source rows through a separately
   bound normalization continuation; keep existing normalized source data intact.
   A parser repair is not training admission or an execution/oracle claim.

This is exact source-format plumbing. Eligibility and scheduling remain the
existing declared data policy. The owner-host diagnostic binds source row11
in seed42/train0: two such headers previously produced a false pending dispatch;
normalization retains all328 messages and matches the genuine subsequent calls.
