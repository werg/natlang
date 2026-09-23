import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkAuthoring } from '../dist/teacher/authoring.js';

const files = {
  'app.ts': 'import type { Ticket } from "./types";\nexport async function urgent(tickets: Ticket[]): Promise<string[]> {\n' +
    '  const flags = await Promise.all(tickets.map(ticket => nl`Does ticket report an outage?`(ticket)));\n' +
    '  return tickets.filter((t, i) => flags[i]).map(t => t.id);\n}\n',
  'types.ts': 'export type Ticket = { id: string, text: string };\n',
};
const spec = { module: 'app.ts', export: 'urgent', requires: { nl: true },
  runs: [{ args: [[{ id: 'a', text: 'site down' }, { id: 'b', text: 'typo' }]], expected: ['a'] }],
  oracle: [{ match: ['"a"'], value: true }, { match: ['"b"'], value: false }] };

test('authored code is judged by running it with an oracle for its nl children', async () => {
  // `import type ... from "./types"` resolves to the folder's aliases, so the child is typed (ticket: Ticket).
  assert.deepEqual(await checkAuthoring(files, spec), { ok: true, problems: [], results: [['a']] });
  const wrong = await checkAuthoring({ ...files, 'app.ts': files['app.ts'].replace('flags[i]', '!flags[i]') }, spec);
  assert.deepEqual(wrong.problems, ['run 0: got ["b"], expected ["a"]']);
  const regex = await checkAuthoring({ ...files, 'app.ts': 'export async function urgent(tickets: { id: string, text: string }[]): Promise<string[]> {\n  return tickets.filter(t => /down/.test(t.text)).map(t => t.id);\n}\n' }, spec);
  assert.deepEqual(regex.problems, ['app.ts creates no nl function']);
});
