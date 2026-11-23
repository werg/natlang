# Runbook Engine (`applications/runbook-engine/`): Decomposition

Tier 2 flagship A7 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3): parses Markdown operational runbooks,
executes their dependency DAG with asymmetric safety gating (`0` unconfirmed destructive actions), tracks idempotent
effects per step ID, and converges post-step health checks.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `engine.ts`):**
  - `parseMarkdownRunbook` & `orderRunbookSteps`: deterministic Markdown parser and topological DAG sorter with
    explicit cycle and missing-dependency detection.
  - `isRoutineReadOnlyCommand` wrapped in `pluggable` (`default: 'crisp'`): settles strictly read-only inspection
    commands (rejecting any shell chaining or mutation verbs) with zero model calls.
  - Asymmetric risk gate: `escalate(assessStepRisk, { to: 'teacher', floor: 0.85, risk: { destructive: 0.2 } })`
    blocks every unconfirmed `destructive` step and skips its downstream dependents.
  - `createOnceOpsService`: declares `[ONCE_EFFECTS]: ['runStep']` and caches completed step results by `stepId`
    so retries never re-run already executed operational commands.
  - `convergeHealthCheck`: polls `service.readHealth(...)` inside `iterateOn(...).withMeasure(...).withLimit(...)`,
    fencing external telemetry with `untrusted(...)` before passing each reading to `checkHealth.nl`.
- **Typed semantic policy (`.nl` leaves):**
  - `assessStepRisk.nl` (`readout: decision`, returns `StepRisk`): classifies step operational impact into
    `'safe' | 'confirm' | 'destructive'`.
  - `checkHealth.nl` (`readout: decision`, returns `boolean`): judges whether external service telemetry satisfies
    the step's recovery target condition.
