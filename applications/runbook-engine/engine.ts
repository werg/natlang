/**
 * Runbook Engine (WS6 Tier 2 flagship A7):
 * - Exact Markdown runbook parser and deterministic topological DAG runner in TypeScript.
 * - Crisp `pluggable` fast-path for strictly read-only inspection steps (0 model calls).
 * - Asymmetric safety gate via `escalate(assessStepRisk, { to: 'teacher', floor: 0.85, risk: { destructive: 0.2 } })`
 *   refusing every unconfirmed `destructive` step (0 unconfirmed destructive actions invariant!).
 * - `ONCE_EFFECTS` idempotency wrapper keyed by step ID so repeated executions never duplicate side effects.
 * - `iterateOn` post-step health check convergence bounded by `withMeasure` and `withLimit`.
 */
import { ONCE_EFFECTS, iterateOn, pluggable, untrusted, type PluggableSetting } from '@natlang/node';
import { escalate } from '@natlang/node/std';
import assessStepRisk from './assessStepRisk.nl';
import checkHealth from './checkHealth.nl';
import type {
  ExecutionReport,
  HealthReading,
  OpsCommandResult,
  OpsService,
  ParsedRunbook,
  RunbookStep,
  StepExecution,
  StepRisk,
} from './types.js';

export type RunbookOptions = {
  /** Pluggable mode for routine read-only step detection (default `'crisp'`). */
  routineMode?: PluggableSetting;
  /** Stronger model name for asymmetric risk escalation (default `'teacher'`). */
  escalateTo?: string;
  /** Confidence floor for `assessStepRisk` (default `0.85`). */
  riskFloor?: number;
  /** Asymmetric risk floor for `'destructive'` escalation (default `0.2`). */
  destructiveRiskFloor?: number;
  /** Set of step IDs explicitly confirmed by a human operator before execution. */
  confirmedStepIds?: ReadonlySet<string>;
  /** Optional interactive confirmation callback; returning false blocks execution. */
  confirmStep?: (step: RunbookStep, risk: StepRisk) => Promise<boolean> | boolean;
  /** Whether `'confirm'`-class steps also require explicit confirmation (default `false`; `'destructive'` ALWAYS requires confirmation). */
  requireConfirmForConfirmSteps?: boolean;
  /** Maximum post-step health check polls per step (default `5`). */
  maxHealthChecks?: number;
};

/** Idempotent operations service declaring ONCE_EFFECTS and tracking executed step IDs. */
export type OnceOpsService = OpsService & {
  readonly [ONCE_EFFECTS]: readonly ['runStep'];
  executedStepCount(stepId: string): number;
};

type HealthLoopState = {
  attempt: number;
  maxAttempts: number;
  readings: HealthReading[];
  recovered: boolean;
};

const CURL_MUTATING_FLAGS =
  /(?:^|\s)(?:-[a-zA-Z]*[XdFToOKbc]|--(?:request|data(?:-[a-z-]+)?|form(?:-string)?|upload-file|output|remote-name(?:-all)?|json|config|cookie(?:-jar)?|next))\b/;

/**
 * Exact crisp detector for strictly read-only diagnostic commands (no shell chaining, redirection, or mutations).
 */
export function isRoutineReadOnlyCommand(command: string): boolean {
  const trimmed = command.trim();
  if (!trimmed) return false;
  if (/[;&|><`$()\n\r\\]/.test(trimmed)) return false;
  if (/\b(rm|drop|delete|truncate|wipe|format|kill|shutdown|reboot|destroy|purge|apply|patch|scale|drain|restart)\b/i.test(trimmed)) {
    return false;
  }
  if (/^curl\b/i.test(trimmed)) {
    if (CURL_MUTATING_FLAGS.test(trimmed)) return false;
    return /^curl\s+-sf?\s+https?:\/\/\S+\/health\S*$/i.test(trimmed);
  }
  return /^(?:kubectl\s+(?:get|describe|logs)\b|systemctl\s+status\b|df\s+-h\b|uptime\b|pg_isready\b|redis-cli\s+ping\b)/i.test(trimmed);
}

/**
 * Exact Markdown runbook parser.
 * Format:
 *   # Runbook Title
 *   ## <step-id>: <Step Title>
 *   Depends: <id1>, <id2>
 *   Health: <target recovery condition>
 *   Confirm: true
 *   Description prose...
 *   ```bash
 *   <command>
 *   ```
 */
export function parseMarkdownRunbook(markdown: string): ParsedRunbook {
  const lines = markdown.replace(/\r\n/g, '\n').split('\n');
  let title = 'Untitled Runbook';
  const steps: RunbookStep[] = [];

  let current: RunbookStep | null = null;
  let inCodeFence = false;
  let commandLines: string[] = [];
  let descLines: string[] = [];

  const flushCurrent = () => {
    if (!current) return;
    current.command = commandLines.join('\n').trim();
    current.description = descLines.join(' ').trim();
    steps.push(current);
    current = null;
    commandLines = [];
    descLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (/^\s*```/.test(line)) {
      inCodeFence = !inCodeFence;
      continue;
    }
    if (inCodeFence) {
      if (current) commandLines.push(rawLine);
      continue;
    }

    const h1 = /^#\s+(.+)$/.exec(line);
    if (h1 && !current && steps.length === 0) {
      title = h1[1]!.trim();
      continue;
    }

    const h2 = /^##\s+([A-Za-z0-9_-]+)\s*:\s*(.+)$/.exec(line);
    if (h2) {
      flushCurrent();
      current = {
        id: h2[1]!.trim(),
        title: h2[2]!.trim(),
        command: '',
        description: '',
        dependsOn: [],
      };
      continue;
    }

    if (!current) continue;
    const trimmed = line.trim();
    if (!trimmed) continue;

    const depMatch = /^Depends:\s*(.+)$/i.exec(trimmed);
    if (depMatch) {
      current.dependsOn = depMatch[1]!
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      continue;
    }

    const healthMatch = /^Health:\s*(.+)$/i.exec(trimmed);
    if (healthMatch) {
      current.healthCheck = healthMatch[1]!.trim();
      continue;
    }

    const confirmMatch = /^Confirm:\s*(true|yes|1)$/i.exec(trimmed);
    if (confirmMatch) {
      current.requiresConfirmation = true;
      continue;
    }

    descLines.push(trimmed);
  }

  flushCurrent();
  return { title, steps };
}

/** Deterministic topological sort over the runbook step DAG; throws on missing dependency or cycle. */
export function orderRunbookSteps(steps: readonly RunbookStep[]): RunbookStep[] {
  const byId = new Map<string, RunbookStep>();
  for (const step of steps) {
    if (byId.has(step.id)) throw new Error(`Duplicate runbook step id "${step.id}".`);
    byId.set(step.id, step);
  }

  for (const step of steps) {
    for (const dep of step.dependsOn) {
      if (!byId.has(dep)) throw new Error(`Runbook step "${step.id}" depends on unknown step "${dep}".`);
    }
  }

  const ordered: RunbookStep[] = [];
  const placed = new Set<string>();
  for (let pass = 0; pass < steps.length; pass++) {
    let progressed = false;
    for (const step of steps) {
      if (placed.has(step.id)) continue;
      if (step.dependsOn.every(dep => placed.has(dep))) {
        ordered.push(step);
        placed.add(step.id);
        progressed = true;
      }
    }
    if (!progressed) break;
  }

  if (ordered.length !== steps.length) {
    throw new Error('Runbook dependency graph contains a cycle.');
  }
  return ordered;
}

/**
 * Wrap an OpsService with `ONCE_EFFECTS` and per-stepId idempotency so re-executing the same step ID
 * returns the earlier result without re-running the underlying operational command.
 */
export function createOnceOpsService(inner: OpsService): OnceOpsService {
  if (ONCE_EFFECTS in inner && typeof (inner as Partial<OnceOpsService>).executedStepCount === 'function') {
    return inner as OnceOpsService;
  }
  // Store the in-flight Promise immediately per stepId so concurrent calls never double-execute.
  const byStepId = new Map<string, Promise<OpsCommandResult>>();
  const execCount = new Map<string, number>();

  return {
    [ONCE_EFFECTS]: ['runStep'] as const,
    async runStep(stepId: string, command: string): Promise<OpsCommandResult> {
      const existing = byStepId.get(stepId);
      if (existing) {
        return { ...(await existing), deduplicated: true };
      }
      execCount.set(stepId, (execCount.get(stepId) ?? 0) + 1);
      const pending = Promise.resolve()
        .then(() => inner.runStep(stepId, command))
        .then(
          result => {
            if (!result.ok) byStepId.delete(stepId);
            return result;
          },
          error => {
            byStepId.delete(stepId);
            throw error;
          },
        );
      byStepId.set(stepId, pending);
      return pending;
    },
    readHealth(stepId: string, attempt: number) {
      return inner.readHealth(stepId, attempt);
    },
    executedStepCount(stepId: string): number {
      return execCount.get(stepId) ?? 0;
    },
  };
}

function fenceHealthReading(reading: HealthReading): HealthReading {
  return {
    stepId: reading.stepId,
    attempt: reading.attempt,
    status: reading.status,
    metrics: { ...reading.metrics },
    message: untrusted(String(reading.message), `ops:${reading.stepId}:health:${reading.attempt}`),
  };
}

/**
 * Poll service health after a step using `iterateOn` bounded by `withMeasure` and `withLimit`
 * until `checkHealth.nl` confirms recovery or attempts are exhausted.
 */
export async function convergeHealthCheck(
  stepId: string,
  targetCondition: string,
  service: OpsService,
  maxAttempts = 5,
): Promise<{ recovered: boolean; readings: HealthReading[] }> {
  const bound = Math.max(1, maxAttempts);
  const initial: HealthLoopState = {
    attempt: 0,
    maxAttempts: bound,
    readings: [],
    recovered: false,
  };

  const finalState = await iterateOn(
    async (state: HealthLoopState): Promise<HealthLoopState> => {
      const attempt = state.attempt + 1;
      const rawReading = await service.readHealth(stepId, attempt);
      const reading = fenceHealthReading(rawReading);
      const recovered = await checkHealth(reading, targetCondition);
      return {
        attempt,
        maxAttempts: state.maxAttempts,
        readings: [...state.readings, reading],
        recovered,
      };
    },
    initial,
  )
    .withMeasure(state => (state.recovered ? 0 : Math.max(0, state.maxAttempts - state.attempt)))
    .withLimit({ maxSteps: bound })
    .until(state => state.recovered || state.attempt >= state.maxAttempts);

  return { recovered: finalState.recovered, readings: finalState.readings };
}

/** Parse and execute a Markdown runbook with asymmetric risk gating, ONCE_EFFECTS idempotency, and health convergence. */
export async function executeRunbook(
  markdown: string,
  service: OpsService,
  options: RunbookOptions = {},
): Promise<ExecutionReport> {
  const parsed = parseMarkdownRunbook(markdown);
  const ordered = orderRunbookSteps(parsed.steps);
  const onceService = ONCE_EFFECTS in service ? (service as OnceOpsService) : createOnceOpsService(service);

  const safeAssessRisk = escalate(assessStepRisk, {
    to: options.escalateTo ?? 'teacher',
    floor: options.riskFloor ?? 0.85,
    risk: { destructive: options.destructiveRiskFloor ?? 0.2 },
  });

  // Pluggable read-only fast path: crisp matches strictly read-only commands (0 model calls); nl asks assessStepRisk.nl
  // whether the step is safe; shadow runs both, records agreement, and serves crisp.
  const routineMode = options.routineMode ?? 'crisp';
  // One risk decision per step, shared by the fast path's nl side and the main gate.
  const assessed = new Map<string, ReturnType<typeof safeAssessRisk>>();
  const assessOnce = (step: RunbookStep) => {
    let pending = assessed.get(step.id);
    if (!pending) assessed.set(step.id, pending = safeAssessRisk(step));
    return pending;
  };
  const routineFastPath = pluggable(
    {
      crisp: (step: RunbookStep): boolean => isRoutineReadOnlyCommand(step.command),
      nl: async (step: RunbookStep): Promise<boolean> => (await assessOnce(step)).value === 'safe',
    },
    routineMode,
    { default: 'crisp', serve: 'crisp', name: 'runbook-routine-readonly-check' },
  );
  const confirmed = options.confirmedStepIds ?? new Set<string>();
  const completedOk = new Set<string>();
  const executions: StepExecution[] = [];
  let unconfirmedDestructiveExecuted = 0;

  for (const step of ordered) {
    const failedDep = step.dependsOn.find(dep => !completedOk.has(dep));
    if (failedDep) {
      executions.push({
        stepId: step.id,
        title: step.title,
        risk: 'safe',
        source: 'crisp',
        confidence: 1,
        escalated: false,
        status: 'skipped-dependency',
        healthChecks: [],
      });
      continue;
    }

    let risk: StepRisk;
    let source: 'crisp' | 'nl';
    let confidence: number;
    let escalated: boolean;
    let escalationReason: string | undefined;

    if (routineMode !== 'nl' && await routineFastPath(step)) {
      risk = 'safe';
      source = 'crisp';
      confidence = 1;
      escalated = false;
    } else {
      const decision = await assessOnce(step);
      risk = decision.value;
      source = 'nl';
      confidence = decision.confidence;
      escalated = decision.escalated;
      escalationReason = decision.reason;
    }
    const needsConfirmation =
      risk === 'destructive' ||
      Boolean(step.requiresConfirmation) ||
      (risk === 'confirm' && Boolean(options.requireConfirmForConfirmSteps));

    let isConfirmed = false;
    if (needsConfirmation) {
      isConfirmed =
        confirmed.has(step.id) || (options.confirmStep ? await options.confirmStep(step, risk) : false);
      if (!isConfirmed) {
        executions.push({
          stepId: step.id,
          title: step.title,
          risk,
          source,
          confidence,
          escalated,
          ...(escalationReason ? { escalationReason } : {}),
          status: 'blocked-unconfirmed',
          healthChecks: [],
        });
        continue;
      }
    }

    if (risk === 'destructive' && !isConfirmed) unconfirmedDestructiveExecuted++;
    const result = await onceService.runStep(step.id, step.command);
    if (!result.ok) {
      executions.push({
        stepId: step.id,
        title: step.title,
        risk,
        source,
        confidence,
        escalated,
        ...(escalationReason ? { escalationReason } : {}),
        status: 'failed',
        output: result.output,
        ...(result.deduplicated ? { deduplicated: true } : {}),
        healthChecks: [],
        recovered: false,
      });
      continue;
    }

    let healthChecks: HealthReading[] = [];
    let recovered: boolean | undefined;
    if (step.healthCheck) {
      const health = await convergeHealthCheck(step.id, step.healthCheck, onceService, options.maxHealthChecks ?? 5);
      healthChecks = health.readings;
      recovered = health.recovered;
    }

    // A step whose health check never converged is not a completed prerequisite: its dependents are skipped.
    if (recovered !== false) completedOk.add(step.id);
    executions.push({
      stepId: step.id,
      title: step.title,
      risk,
      source,
      confidence,
      escalated,
      ...(escalationReason ? { escalationReason } : {}),
      status: 'executed',
      output: result.output,
      ...(result.deduplicated ? { deduplicated: true } : {}),
      healthChecks,
      ...(recovered !== undefined ? { recovered } : {}),
    });
  }

  const healthCheckedSteps = executions.filter(s => s.recovered !== undefined);
  const allRecovered =
    executions.every(s => s.status === 'executed') &&
    healthCheckedSteps.every(s => s.recovered === true);

  return {
    runbookTitle: parsed.title,
    totalSteps: ordered.length,
    executedSteps: executions.filter(s => s.status === 'executed').length,
    blockedSteps: executions.filter(s => s.status === 'blocked-unconfirmed').length,
    unconfirmedDestructiveExecuted,
    recovered: allRecovered,
    steps: executions,
  };
}
