import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

test('the triage example classifies in parallel, counts exactly, and shortens its summary with iterateOn', async () => {
  const outDir = mkdtempSync(join(tmpdir(), 'natlang-triage-'));
  const built = buildProject({ project: fileURLToPath(new URL('../../examples/triage', import.meta.url)), outDir, runtimeModule,
    writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const { triage } = await import(pathToFileURL(join(outDir, 'triage.js')).href);
  const tickets = [
    'I was charged twice for my March invoice.',
    'CHEAP WATCHES!!! Visit our store today for 90% off.',
    'PRODUCTION IS DOWN. Customers cannot log in at all.',
    'Export to CSV produces an empty file, no rush.',
    'Payments are failing for every customer right now.',
  ];
  const model = scriptedModel(opening => {
    if (opening.includes('Pick the label')) return 'result = /WATCHES/.test(ticket) ? "spam" : /charged|Payments/.test(ticket) ? "billing" : "technical"';
    if (opening.includes('attention within the hour') || opening.includes('A ticket is urgent')) return 'result = /DOWN|Payments are failing/.test(ticket)';
    if (opening.includes('Write one paragraph')) return 'result = (tickets.join(" ") + " ").repeat(4).trim()';
    if (opening.includes('noticeably shorter')) return 'result = text.split(/\\s+/).slice(0, Math.ceil(text.split(/\\s+/).length / 2)).join(" ")';
    return null;
  });
  const report = await createNatlangRuntime({ model: model.driver }).run(() => triage(tickets, 'billing, technical, spam'));
  assert.equal(report.urgent, 2);
  assert.deepEqual(report.by_label, { billing: 2, spam: 1, technical: 2 });
  assert.match(report.summary, /cannot log in/);
  assert.ok(report.summary.split(/\s+/).length <= 60, report.summary);
  assert.ok(model.openings.filter(opening => opening.includes('noticeably shorter')).length >= 1);
});
