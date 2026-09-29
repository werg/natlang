import { defineEvaluationSuite } from '@natlang/node/evaluation';
import { executorIdentityForChoice, loadModelConfiguration } from '@natlang/node/model';
export default defineEvaluationSuite({
  id: 'adaptation-moderation-v1', program: { root: '.', id: 'moderation', entry: 'main.ts' },
  components: 'joint', cases: './cases.jsonl', models: { executor: 'default', reflection: 'default' },
  executorIdentity: executorIdentityForChoice(loadModelConfiguration('default').choice),
  policy: { codeEdits: 'deny', settings: { maxTurns: 5, turnTokens: 384, maxSeconds: 30, contextTokens: 8192 } },
  budget: { maxRollouts: 40, maxProposals: 3, maxModelCalls: 260 },
  fixture: { async create(testCase, context) {
    const app = await context.loadFreshProgram() as { review(message: unknown): Promise<unknown>;
      inspectRisk(message: unknown): Promise<'safe' | 'unsafe'>; inspectDecision(risk: 'safe' | 'unsafe'): Promise<boolean> };
    let localRisk: 'safe' | 'unsafe' | null = null, localDecision: boolean | null = null;
    const failures: string[] = [];
    return {
      execute: () => context.runtime.run(async () => {
        try { localRisk = await app.inspectRisk(testCase.input); } catch (error) { failures.push(String(error)); }
        try { localDecision = await app.inspectDecision(localRisk ?? 'unsafe'); } catch (error) { failures.push(String(error)); }
        return await app.review(testCase.input);
      }),
      observe: (result, outcome) => ({ result: outcome.kind === 'returned' ? result : null, localRisk, localDecision, failures }),
    };
  } },
  score(testCase, observation: unknown, outcome) {
    const observed = observation as { result?: { risk: string; publish: boolean }; localRisk: string | null; localDecision: boolean | null; failures: string[] };
    const actual = observed.result;
    const expected = testCase.expected as { risk: string; publish: boolean };
    const risk = actual?.risk === expected.risk, publish = actual?.publish === expected.publish;
    const localRisk = observed.localRisk === expected.risk;
    const localDecision = observed.localRisk !== null && observed.localDecision === (observed.localRisk === 'safe');
    return { quality: (Number(localRisk) + Number(localDecision) + Number(risk) + Number(publish)) / 4,
      metrics: { localRisk: Number(localRisk), localDecision: Number(localDecision), finalRisk: Number(risk), finalDecision: Number(publish) },
      gates: { completed: outcome.kind === 'returned' && observed.failures.length === 0,
        consistent: actual ? actual.publish === (actual.risk === 'safe') : false },
      feedback: `Helper risk ${localRisk ? 'correct' : 'incorrect'}; helper decision ${localDecision ? 'correct' : 'incorrect'}; root risk ${risk ? 'correct' : 'incorrect'}; final decision ${publish ? 'correct' : 'incorrect'}.` };
  },
  feedback(testCase, result) { return { input: testCase.input, actual: result.outcome.value ?? null }; },
});
