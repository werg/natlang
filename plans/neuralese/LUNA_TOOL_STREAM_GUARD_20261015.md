# Luna tool-call stream guard (2026-10-15)

Seed 52068, worker 03, case 012 (`policy_after_measure:latency3:strict`) exposed
a measured provider-output failure. Its second Luna request streamed more than
114,000 tool-call argument deltas and 374,732 bytes over 737 seconds, while
producing no text, completed reply, saved turn or result. The source task is a
small p95 policy decision. This is not evidence that its answer is hard or
wrong. Pop stopped only the collector child PID 3054751 to release the worker;
the next case proceeded. The zero-result attempt is explicitly recorded in
the seed 52068 `reviewed-exclusions.json`, bound to its empty output SHA-256.
It is excluded from SFT and DPO. A prior seed 52021 case had the same pattern:
99,651 tool-call deltas and 242,776 bytes until the 1,799-second case budget.

`scripts/run_bonsai_queue.py` now records the largest tool-call stream byte
count in its content-free provider observations. For each action it derives a
guard from the requested output ceiling: `max(131072, 32 * action_max_tokens)`
bytes. Crossing it before a reply terminates that exact collector case with
`provider_stream_overflow` and keeps the zero-result failure explicit for
review. Ordinary reasoning deltas, a legitimate rate-limit wait, and elapsed
time alone do not trigger it. This guard is a resource stopgap for observed
runaway argument streams; the upstream reason that Codex Responses can stream
far beyond the requested `max_output_tokens` remains to be localized. In
particular, check whether Pi emits cumulative revisions as deltas or whether
the provider ignores the ceiling. The collection observation policy is now
version 2. New workers use the change; already running workers retain their
loaded supervisor code.
