/** `natlang run applications/runbook-engine -- [INPUT.json]` (or stdin JSON). */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { untrusted, type TargetContext } from '@natlang/node';
import { createOnceOpsService, executeRunbook, type RunbookOptions } from './engine.js';
import type { HealthReading, OpsCommandResult, OpsService } from './types.js';

type CliPayload = {
  markdown: string;
  confirmedStepIds?: string[];
  healthByStep?: Record<string, HealthReading[]>;
  options?: Pick<RunbookOptions, 'routineMode' | 'escalateTo' | 'riskFloor' | 'destructiveRiskFloor' | 'requireConfirmForConfirmSteps' | 'maxHealthChecks'>;
};

function createSimulatedOpsService(healthByStep: Record<string, HealthReading[]> = {}): OpsService {
  return {
    runStep: async (stepId, command): Promise<OpsCommandResult> => ({
      stepId,
      command,
      ok: true,
      output: `executed ${stepId}: ${command}`,
    }),
    readHealth: async (stepId, attempt): Promise<HealthReading> => {
      const series = healthByStep[stepId];
      if (series && series.length > 0) {
        const entry = series[Math.min(attempt - 1, series.length - 1)]!;
        return { ...entry, stepId, attempt, message: untrusted(String(entry.message), `ops:${stepId}:health:${attempt}`) };
      }
      return {
        stepId,
        attempt,
        status: 'healthy',
        metrics: { errorRate: 0 },
        message: untrusted(`step ${stepId} healthy on attempt ${attempt}`, `ops:${stepId}:health:${attempt}`),
      };
    },
  };
}

export async function main(context: TargetContext): Promise<number> {
  const args = (context.arguments ?? {}) as { file?: string };
  const path = args.file ?? context.args[0];
  const raw = path ? readFileSync(resolve(context.workspace, path), 'utf8') : readFileSync(0, 'utf8');
  const payload = JSON.parse(raw) as CliPayload;
  const service = createOnceOpsService(createSimulatedOpsService(payload.healthByStep));
  const report = await context.runtime.run(() =>
    executeRunbook(payload.markdown, service, {
      ...(payload.options ?? {}),
      ...(payload.confirmedStepIds ? { confirmedStepIds: new Set(payload.confirmedStepIds) } : {}),
    }),
  );
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return 0;
}
