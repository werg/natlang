import { defineEvaluationSuite } from '@natlang/node/evaluation';
import { executorIdentityForChoice, loadModelConfiguration } from '@natlang/node/model';
import { Folder } from '@natlang/node';
export default defineEvaluationSuite({
  id: 'adaptation-stateful-v1', program: { root: '.', id: 'stateful', entry: 'main.ts' },
  components: 'lambda-only', cases: './cases.jsonl', models: { executor: 'default', reflection: 'default' },
  executorIdentity: executorIdentityForChoice(loadModelConfiguration('default').choice),
  policy: { codeEdits: 'deny', settings: { maxTurns: 5, turnTokens: 384, maxSeconds: 30, contextTokens: 8192 } },
  budget: { maxRollouts: 24, maxProposals: 3, maxModelCalls: 120 },
  fixture: { async create(testCase, context) {
    const app = await context.loadFreshProgram() as { append: Parameters<Folder['apply']>[0] };
    const folder = Folder.fromFiles({ 'log.txt': 'start\n', 'untouched.txt': 'keep' });
    return { execute: () => context.runtime.run(() => folder.apply(app.append, testCase.input)),
      observe: async result => ({ ...(result === undefined ? {} : { result }), log: await folder.readText('log.txt'), untouched: await folder.readText('untouched.txt') }) };
  } },
  score(testCase, observation: unknown, outcome) {
    const actual = observation as { result?: string; log: string; untouched: string };
    return { quality: actual.log === testCase.expected && actual.result === 'ok' ? 1 : 0,
      gates: { completed: outcome.kind === 'returned', preservesOtherFiles: actual.untouched === 'keep' },
      feedback: actual.log === testCase.expected ? 'Append is correct.' : 'Existing text must be preserved and exactly one newline-terminated note appended.' };
  },
  feedback(testCase, result) { return { note: testCase.input, actual: result.outcome.value ?? null }; },
});
