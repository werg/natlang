import { defineEvaluationSuite } from '@natlang/node/evaluation';
import { executorIdentityForChoice, loadModelConfiguration } from '@natlang/node/model';
export default defineEvaluationSuite({
  id: 'adaptation-triage-v1', program: { root: '.', id: 'triage', entry: 'main.ts' },
  components: 'lambda-only', cases: './cases.jsonl', models: { executor: 'default', reflection: 'default' },
  executorIdentity: executorIdentityForChoice(loadModelConfiguration('default').choice),
  policy: { codeEdits: 'deny', settings: { maxTurns: 4, turnTokens: 256, maxSeconds: 30, contextTokens: 8192 } },
  budget: { maxRollouts: 48, maxProposals: 3, maxModelCalls: 240 },
  fixture: { async create(testCase, context) {
    const app = await context.loadFreshProgram() as { triage(message: unknown): Promise<string> };
    return { execute: () => context.runtime.run(() => app.triage(testCase.input)) };
  } },
  score(testCase, observation: unknown, outcome) {
    const result = (observation as { result?: string }).result;
    return { quality: result === testCase.expected ? 1 : 0,
      gates: { completed: outcome.kind === 'returned' }, feedback: result === testCase.expected ? 'Correct.' : `Expected ${testCase.expected}; observed ${result ?? outcome.error?.outcome}.` };
  },
  feedback(testCase, result) { return { input: testCase.input, actual: result.outcome.value ?? null }; },
});
