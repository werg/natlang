# Repetition-guard progress polling decomposition (2026-10-23)

The collector already records each completed response in an atomically replaced, durable `.partial.json` before it requests the next turn. The queue supervisor currently waits up to thirty seconds before it reads that file and evaluates its configured exact action/observation repetition limit. A short limit can therefore permit many extra model turns after the evidence is already on disk.

The correction keeps the existing decision rule and terminal evidence. The queue supervisor watches the exact per-job partial files using their filesystem identity, size, and nanosecond modification time. A changed identity triggers the existing partial validator and exact repetition predicate; unchanged files skip those reads. With a repetition limit enabled, the child wait is bounded to a short polling interval, while the ordinary thirty-second activity heartbeat remains the cadence for provider-stream, decode-progress, and silence accounting. With the limit disabled, the existing thirty-second child wait remains unchanged.

| Part | Representation | Reason |
| --- | --- | --- |
| Whether a repeated action and observation meet the configured bound | Existing crisp verifier in `identical_turn_streak` | It compares exact recorded actions, observations, and invocation identity and produces the stop receipt. |
| Whether the durable partial changed | Crisp filesystem metadata snapshot | Atomic rename, inode, size, and `mtime_ns` are external filesystem facts; no policy is inferred from them. |
| When to re-check the existing rule | Crisp bounded supervisor wait and progress gate | It schedules exact verification promptly without changing the threshold or trainer-facing prompts. |
| Stop classification and saved evidence | Existing queue error/finish journal path | The current `repetition` reason and `limit_evidence` remain authoritative. |

No model-facing text, sampling settings, task scorer, whole-graph capture, thresholds, source data, or accepted training outcomes change. An unchanged-file wake must not re-parse partials; a file update must trigger the same verifier before another long wait. Tests simulate collector atomic replacements arriving rapidly, no-change polling, and terminal guard evidence. EOF/normal child completion and other configured monitoring continue through their existing paths.
