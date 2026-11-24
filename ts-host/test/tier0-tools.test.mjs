// End-to-end tests for Tier 0 campaign tools: applications/idiom-reviewer (WS3 I2).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const runtimeModule = {
  url: new URL('../dist/index.js', import.meta.url).href,
  path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)),
  specifiers: ['@natlang/node'],
};

async function buildApp(name) {
  const outDir = mkdtempSync(join(tmpdir(), `natlang-tier0-${name}-`));
  const built = buildProject({ project: join(repo, 'applications', name), outDir, runtimeModule, writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  return async mod => import(pathToFileURL(join(outDir, mod)).href);
}

const textOf = req => req.messages.flatMap(m => [
  typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
  ...(m.tool_calls ?? []).map(c => c.function?.arguments ?? ''),
]).join('\n');

test('applications/idiom-reviewer: stepKind per step (escalated when unsure), prose interpreter, cue words, untyped external text; idiomatic file clean', async () => {
  const load = await buildApp('idiom-reviewer');
  const { reviewProgramIdioms } = await load('review.js');
  const { programFiles } = await load('main.js');

  const files = [
    {
      path: 'legacy/triage.nl',
      content: [
        '---',
        'args:',
        '  ticket: string',
        '  rubric: string',
        'returns: string',
        '---',
        '1. Sort all words alphabetically and count how many times the keyword "refund" appears.',
        '2. If the word "urgent" is present return high, otherwise return low.',
      ].join('\r\n'),
    },
    {
      path: 'clean/classify.nl',
      content: [
        '---',
        'args:',
        '  ticket: Untrusted<string>',
        'returns: "billing" | "technical"',
        'readout: decision',
        '---',
        'Classify the underlying issue described in ticket.',
      ].join('\n'),
    },
  ];

  const wireOf = req => req.messages.filter(m => m.role === 'tool').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
  const asked = { stepKind: 0, interpreter: 0, cue: 0, external: [] };
  const local = async () => ({ text: '"judgment"', calls: [] });
  local.decide = async req => {
    const prompt = textOf(req), wire = wireOf(req);
    const pick = p => ({ log_probs: req.options.map(o => Math.log(p[JSON.parse(o)] ?? 1e-6)) });
    if (req.options.includes('"guard-for-type"')) {
      asked.stepKind++;
      if (/Sort all words/.test(wire)) return pick({ judgment: 0.04, mechanism: 0.9, 'guard-for-type': 0.03, redundant: 0.03 });
      if (/urgent/.test(wire)) return pick({ judgment: 0.45, mechanism: 0.45, 'guard-for-type': 0.05, redundant: 0.05 }); // unsure
      return pick({ judgment: 0.94, mechanism: 0.02, 'guard-for-type': 0.02, redundant: 0.02 });
    }
    if (/algorithm written in English/.test(prompt)) { asked.interpreter++; return pick({ true: 0.9, false: 0.1 }); }
    if (/cue-word/.test(prompt)) { asked.cue++; return pick(/urgent/.test(wire) ? { true: 0.9, false: 0.1 } : { true: 0.05, false: 0.95 }); }
    if (/Untrusted/.test(prompt)) {
      const name = /"paramName":\s*"(\w+)"|paramName[^\w]+(\w+)/.exec(wire);
      asked.external.push(name?.[1] ?? name?.[2]);
      return pick(/ticket/.test(wire) ? { true: 0.9, false: 0.1 } : { true: 0.1, false: 0.9 });
    }
    throw new Error(`unexpected decision: ${prompt.slice(0, 200)}`);
  };
  const teacherCalls = [];
  const teacher = async () => ({ text: '"mechanism"', calls: [] });
  teacher.decide = async req => {
    teacherCalls.push(textOf(req));
    return { log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'mechanism' ? 0.97 : 0.01)) };
  };

  const report = await createNatlangRuntime({ model: local, models: { teacher } }).run(() => reviewProgramIdioms(files));

  assert.equal(report.filesReviewed, 2);
  assert.equal(report.stepsReviewed, 3, 'CRLF source split into its two numbered steps, plus one paragraph');
  assert.ok(asked.stepKind >= 3);
  assert.ok(asked.interpreter >= 1);
  assert.ok(!report.issues.some(i => i.kind === 'prose-interpreter' && i.file === 'clean/classify.nl'), 'a one-step function is not judged as a whole');
  const kinds = report.issues.filter(i => i.file === 'legacy/triage.nl').map(i => `${i.kind}:${i.stepIndex ?? '-'}:${i.escalated}`).sort();
  assert.deepEqual(kinds, ['cue-word-rule:-:false', 'mechanism-in-prose:0:false', 'mechanism-in-prose:1:true', 'prose-interpreter:-:false',
    'untyped-external-input:-:false']);
  assert.ok(report.issues.some(i => i.kind === 'untyped-external-input' && i.excerpt === 'ticket: string'));
  assert.ok(!report.issues.some(i => i.file === 'clean/classify.nl'), 'the idiomatic file reports no issues');
  assert.ok(teacherCalls.length >= 1 && teacherCalls.every(text => /urgent/.test(text)), 'only the unsure step escalates');
  assert.equal(report.clean, false);

  // main.ts reads a program directory: every .nl under it, skipping build output, with its helper folder noted.
  const dir = mkdtempSync(join(tmpdir(), 'natlang-idiom-dir-'));
  mkdirSync(join(dir, 'route'), { recursive: true });
  mkdirSync(join(dir, 'node_modules/x'), { recursive: true });
  writeFileSync(join(dir, 'route.nl'), files[1].content);
  writeFileSync(join(dir, 'route/helper.ts'), 'export default () => 1;\n');
  writeFileSync(join(dir, 'route/inner.nl'), files[1].content);
  writeFileSync(join(dir, 'node_modules/x/skip.nl'), files[0].content);
  assert.deepEqual(programFiles(dir).map(f => [f.path, f.hasHelpers]), [['route/inner.nl', false], ['route.nl', true]]);
});
