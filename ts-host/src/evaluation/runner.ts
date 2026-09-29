import { ADAPTATION_SCHEMA, type Candidate, type AdaptationArtifact } from '../adaptation/types.js';
import { validateCandidate } from '../adaptation/compatibility.js';
import { artifactDigest } from '../adaptation/schema.js';
import { fingerprint } from '../adaptation/identity.js';
import { summarize } from './metrics.js';
import { runWorker } from './worker.js';
import { UsageGateway, BudgetExhausted, type BudgetLedger } from './usage.js';
import type { PreparedSuite, EvaluateOptions, EvaluationBatch, EvaluationResult } from './types.js';
export class EvaluationInfrastructureError extends Error {
  constructor(readonly caseId: string, readonly attempt: number, cause: unknown, readonly ledger: BudgetLedger) {
    super('evaluation infrastructure failure for ' + caseId + ': ' + (cause instanceof Error ? cause.message : String(cause)), { cause });
    this.name = 'EvaluationInfrastructureError';
  }
}
/** Import/create/dispose every fixture without executing or scoring paid inference. */
export async function preflightEvaluation(prepared: PreparedSuite): Promise<{ cases: number; suiteHash: string }> {
  const gateway = new UsageGateway({ maxRollouts: prepared.cases.length, maxProposals: 0, maxModelCalls: 0 });
  for (const testCase of prepared.cases) {
    await runWorker({ moduleURL: prepared.moduleURL, compiledEntry: prepared.compiledEntry, program: prepared.program,
      testCase, artifact: null, seed: 0, replicate: 0, id: 'preflight:' + testCase.id, preflight: true },
    () => { throw new Error('paid inference is unavailable during fixture preflight'); }, gateway);
  }
  return { cases: prepared.cases.length, suiteHash: prepared.suiteHash };
}
export function candidateArtifact(prepared: PreparedSuite, candidate: Candidate, runId = 'evaluation'): AdaptationArtifact {
  const components = prepared.program.components.filter(component => prepared.components.includes(component.key));
  const validated = validateCandidate(candidate, components);
  const artifact: AdaptationArtifact = { schema: ADAPTATION_SCHEMA, digest: '', program: { id: prepared.program.id,
    buildHash: prepared.program.buildHash, protocol: prepared.program.protocol, guidanceScope: prepared.program.guidanceScope },
    executor: prepared.suite.executorIdentity, policy: prepared.suite.policy ?? { codeEdits: 'deny', settings: {} },
    components: components.map(component => ({ key: component.key, baselineHash: component.baselineHash, contractHash: component.contractHash, value: validated[component.key]! })),
    provenance: { runId, strategy: 'evaluation', engine: 'natlang/v1', suiteHash: prepared.suiteHash, seed: 0, promotion: 'selected', evidence: {} } };
  artifact.digest = artifactDigest(artifact); return artifact;
}
export async function evaluate(prepared: PreparedSuite, options: EvaluateOptions): Promise<EvaluationBatch> {
  const cases = prepared.cases.filter(testCase => options.caseIds ? options.caseIds.includes(testCase.id) : testCase.split === (options.split ?? 'validation'));
  if (!cases.length || options.caseIds?.some(id => !cases.some(testCase => testCase.id === id))) throw new Error('requested evaluation cases are missing');
  const candidate = options.candidate;
  const artifact = candidate ? candidateArtifact(prepared, candidate, options.runId) : null;
  const gateway = options.gateway ?? new UsageGateway(prepared.suite.budget); const results: EvaluationResult[] = [];
  for (const testCase of cases.sort((a, b) => a.id.localeCompare(b.id))) {
    const seed = parseInt(fingerprint({ seed: options.seed ?? 0, case: testCase.id, replicate: options.replicate ?? 0 }).slice(0, 8), 16) >>> 0;
    const id = fingerprint({ suite: prepared.suiteHash, case: testCase.id, candidate: candidate ?? null, seed, replicate: options.replicate ?? 0, policy: prepared.suite.policy ?? null });
    let attempt = 0;
    while (true) {
      gateway.reserve('rollouts', 1, options.signal);
      try { results.push(await runWorker({ moduleURL: prepared.moduleURL, compiledEntry: prepared.compiledEntry, program: prepared.program,
        testCase, artifact, seed, replicate: options.replicate ?? 0, id }, options.driver, gateway, options.signal, options.timeoutMs, options.judge)); break; }
      catch (error) {
        if (error instanceof BudgetExhausted || options.signal?.aborted) throw error;
        if (attempt++ < (options.retries ?? 0)) continue;
        throw new EvaluationInfrastructureError(testCase.id, attempt, error, gateway.snapshot());
      }
    }
  }
  return summarize(results);
}
