/** Finite operational risk classification for a runbook step. */
export type StepRisk = 'safe' | 'confirm' | 'destructive';

/** One parsed step node in an operational Markdown runbook DAG. */
export type RunbookStep = {
  id: string;
  title: string;
  command: string;
  description: string;
  dependsOn: string[];
  healthCheck?: string;
  requiresConfirmation?: boolean;
};

/** Parsed runbook document with title and ordered step nodes. */
export type ParsedRunbook = {
  title: string;
  steps: RunbookStep[];
};

/** One post-step health observation; telemetry message comes from external runtime output. */
export type HealthReading = {
  stepId: string;
  attempt: number;
  status: 'healthy' | 'degraded' | 'failing';
  metrics: Record<string, number>;
  message: Untrusted<string>;
};

/** Result of executing one idempotent operations command. */
export type OpsCommandResult = {
  stepId: string;
  command: string;
  ok: boolean;
  output: string;
  deduplicated?: boolean;
};

/** External operations service boundary executing idempotent ONCE_EFFECTS commands and reading health telemetry. */
export type OpsService = {
  runStep(stepId: string, command: string): Promise<OpsCommandResult> | OpsCommandResult;
  readHealth(stepId: string, attempt: number): Promise<HealthReading> | HealthReading;
};

/** Outcome record for one runbook step. */
export type StepExecution = {
  stepId: string;
  title: string;
  risk: StepRisk;
  source: 'crisp' | 'nl';
  confidence: number;
  escalated: boolean;
  escalationReason?: string;
  status: 'executed' | 'blocked-unconfirmed' | 'skipped-dependency' | 'failed';
  output?: string;
  deduplicated?: boolean;
  healthChecks: HealthReading[];
  recovered?: boolean;
};

/** Aggregate execution report for a runbook run. */
export type ExecutionReport = {
  runbookTitle: string;
  totalSteps: number;
  executedSteps: number;
  blockedSteps: number;
  unconfirmedDestructiveExecuted: number;
  recovered: boolean;
  steps: StepExecution[];
};
