// End-to-end tests for Tier 0 campaign tools: applications/idiom-reviewer (WS3 I2) and applications/task-author (WS3 T3).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync } from 'node:fs';
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

test('applications/idiom-reviewer classifies mechanism/prose-interpreter, cue words, and untyped external inputs with escalation', async () => {
  const load = await buildApp('idiom-reviewer');
  const { reviewProgramIdioms } = await load('review.js');

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
        'Step 1: Sort all words alphabetically and count how many times the keyword "refund" appears.',
        '',
        'If the word "urgent" is present return high, otherwise return low.',
      ].join('\n'),
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

  const driver = async () => ({ text: '"judgment"', calls: [] });
  driver.decide = async req => {
    const prompt = textOf(req);
    if (req.options.includes('"prose-interpreter"')) {
      const wire = req.messages.filter(m => m.role === 'tool').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
      const p = /Step 1: Sort all words/.test(wire)
        ? { judgment: 0.05, mechanism: 0.10, 'prose-interpreter': 0.85 }
        : /urgent/.test(wire)
          ? { judgment: 0.45, mechanism: 0.50, 'prose-interpreter': 0.05 } // unsure -> escalates to teacher
          : { judgment: 0.92, mechanism: 0.05, 'prose-interpreter': 0.03 };
      return { log_probs: req.options.map(o => Math.log(p[JSON.parse(o)])) };
    }
    if (/brittle keyword or cue-word/.test(prompt)) {
      const wire = req.messages.filter(m => m.role === 'tool').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
      const cue = /urgent/.test(wire);
      return { log_probs: req.options.map(o => Math.log((JSON.parse(o) === cue) ? 0.9 : 0.1)) };
    }
    if (/Untrusted/.test(prompt)) {
      const wire = req.messages.filter(m => m.role === 'tool').map(m => typeof m.content === 'string' ? m.content : JSON.stringify(m.content)).join('\n');
      const ext = /"ticket"/.test(wire) && !/"rubric"/.test(wire);
      return { log_probs: req.options.map(o => Math.log((JSON.parse(o) === ext) ? 0.9 : 0.1)) };
    }
    return { log_probs: req.options.map(() => Math.log(1 / req.options.length)) };
  };

  const teacherCalls = [];
  const teacher = async req => { teacherCalls.push(textOf(req)); return { text: '"mechanism"', calls: [] }; };
  teacher.decide = async req => {
    teacherCalls.push(textOf(req));
    return { log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'mechanism' ? 0.95 : 0.025)) };
  };

  const report = await createNatlangRuntime({ model: driver, models: { teacher } })
    .run(() => reviewProgramIdioms(files));

  assert.equal(report.filesReviewed, 2);
  assert.equal(report.groupsReviewed, 3);
  assert.equal(report.clean, false);
  assert.ok(report.issues.some(i => i.file === 'legacy/triage.nl' && i.kind === 'prose-interpreter'));
  assert.ok(report.issues.some(i => i.file === 'legacy/triage.nl' && i.kind === 'mechanism-in-prose' && i.escalated));
  assert.ok(report.issues.some(i => i.file === 'legacy/triage.nl' && i.kind === 'cue-word-rule'));
  assert.ok(report.issues.some(i => i.file === 'legacy/triage.nl' && i.kind === 'untyped-external-input' && i.excerpt.includes('ticket')));
  assert.ok(!report.issues.some(i => i.file === 'clean/classify.nl'), 'idiomatic file reports zero issues');
  assert.ok(teacherCalls.length >= 1);
});

test('applications/task-author repairs structural/baseline issues via semanticRepair and gates on semantic + oracle verdicts', async () => {
  const load = await buildApp('task-author');
  const { authorAndQualifyTask } = await load('author.js');

  let draftAttempt = 0;
  const model = scriptedModel(opening => {
    if (!opening.includes('Author a realistic Semantic-FP task')) return null;
    draftAttempt++;
    if (draftAttempt === 1) {
      // First draft omits verifier & natlang.json -> triggers semanticRepair!
      return `return {
        id: seed.id,
        split_group: seed.splitGroup,
        constructs: [seed.construct],
        oracle: 'verifier',
        prompt: 'Verify PR diffs against team rules with grounded quotes.',
        files: { 'review.ts': 'export {};' }
      }`;
    }
    // Second draft fixes all reported problems!
    return `return {
      id: seed.id,
      split_group: seed.splitGroup,
      constructs: [seed.construct],
      oracle: 'verifier',
      verifier: 'proper_score.mjs',
      prompt: 'Review untrusted PR hunks against security invariants and escalate uncertain block risks.',
      files: { 'natlang.json': '{}', 'review.ts': 'export {};' }
    }`;
  });

  const driver = async req => req.tools.length ? model.driver(req) : { text: 'true', calls: [] };
  driver.decide = async req => ({ log_probs: req.options.map(o => Math.log(JSON.parse(o) === true ? 0.92 : 0.08)) });

  const seed = {
    id: 'sfp-pr-001',
    domain: 'code-review',
    construct: 'escalate',
    oracleKind: 'verifier',
    summary: 'Escalate risky PR merge verdicts under asymmetric block floor.',
    splitGroup: 'sfp:pr:001',
  };

  const result = await createNatlangRuntime({ model: driver, models: { teacher: driver } })
    .run(() => authorAndQualifyTask(seed, { reservedGroups: new Set(['sfp:held:999']) }));

  assert.equal(result.admitted, true);
  assert.equal(draftAttempt, 2, 'semanticRepair fixed the missing natlang.json and verifier');
  assert.deepEqual(result.structuralIssues, []);
  assert.equal(result.semanticVerdict, true);
  assert.equal(result.oracleVerdict, true);
  assert.equal(result.baselineDefeated, true);
});
