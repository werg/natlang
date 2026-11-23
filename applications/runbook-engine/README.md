# Runbook Engine

Tier 2 Semantic-FP flagship application (`A7`) for executing Markdown operational runbooks with asymmetric safety gates and health-check convergence.

## Highlights

- **Exact Markdown DAG execution**: Parses runbook headings, dependencies, fenced commands, and recovery conditions into a topologically ordered step graph in TypeScript.
- **Zero unconfirmed destructive actions (`escalate`)**: Asymmetric risk floor (`risk: { destructive: 0.2 }`, `floor: 0.85`) escalates any step with non-trivial destructive probability to `teacher` and refuses execution unless explicitly confirmed.
- **Routine read-only crisp fast-path (`pluggable`)**: Pure read-only status probes settle crisply with 0 LLM calls; any chained or mutating command goes through the safety gate.
- **Idempotent execution (`ONCE_EFFECTS`)**: `createOnceOpsService` tracks step IDs so repeated invocations never duplicate side effects.
- **Bounded health convergence (`iterateOn` + `withMeasure`)**: Polls post-step telemetry fenced as `Untrusted<string>` until `checkHealth.nl` confirms recovery or the step limit is reached.
