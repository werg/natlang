// natlang check --idiom runs the idiom reviewer (applications/idiom-reviewer, a natlang program) through `natlang run`
// and reports its findings as diagnostics: warnings, or errors under --idiom-strict when the reviewer is confident.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const CLI = fileURLToPath(new URL('../dist/cli/main.js', import.meta.url));
const natlang = (args, env = {}) => spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8', env: { ...process.env, NO_COLOR: '1', NATLANG_SERVER: 'http://127.0.0.1:9', ...env } });

function program() {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-check-idiom-'));
  writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'idiom-target', version: '0.1.0' }));
  writeFileSync(join(dir, 'route.nl'), '---\nargs:\n  ticket: string\nreturns: string\n---\nRead ticket.\nSplit ticket on commas and count the parts.\n');
  return dir;
}

/** A stand-in reviewer with the real one's interface (`natlang run DIR -- PROGRAM`, JSON report on stdout). */
function reviewer(issues) {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-idiom-reviewer-'));
  writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: '@natlang/idiom-reviewer', version: '0.1.0',
    include: ['main.ts'], targets: { 'idiom-reviewer': { entry: 'main.ts', authority: ['filesystem:workspace'] } } }));
  writeFileSync(join(dir, 'main.ts'), `export async function main(context: { args: string[]; io: { output: { write(text: string): void } } }): Promise<number> {
  const issues = ${JSON.stringify(issues)};
  context.io.output.write(JSON.stringify({ filesReviewed: 1, stepsReviewed: 2, clean: !issues.length, issues, target: context.args[0] }) + '\\n');
  return issues.length ? 1 : 0;
}
`);
  return dir;
}

const ISSUE = { file: 'route.nl', kind: 'mechanism-in-prose', stepIndex: 1, excerpt: 'Split ticket on commas and count the parts.',
  confidence: 0.91, escalated: false, suggestion: 'Move this step into a TypeScript helper and keep the `.nl` function for the judgment.' };

test('natlang check --idiom reports the reviewer\'s findings as diagnostics at their lines', () => {
  const dir = program(), fake = reviewer([ISSUE]);
  try {
    const ran = natlang(['check', dir, '--idiom', '--json'], { NATLANG_IDIOM_REVIEWER: fake });
    assert.equal(ran.status, 0, ran.stderr);
    const report = JSON.parse(ran.stdout);
    const found = report.diagnostics.find(item => item.code === 'idiom-review-mechanism-in-prose');
    assert.ok(found, JSON.stringify(report.diagnostics));
    assert.equal(found.severity, 'warning');
    assert.equal(found.file, 'route.nl');
    assert.equal(found.line, 7);
    assert.match(found.message, /Move this step into a TypeScript helper/);

    const strict = natlang(['check', dir, '--idiom-strict', '--json'], { NATLANG_IDIOM_REVIEWER: fake });
    assert.equal(strict.status, 1);
    const error = JSON.parse(strict.stdout).diagnostics.find(item => item.code === 'idiom-review-mechanism-in-prose');
    assert.ok(error && error.severity === undefined, 'an error (check --json marks only warnings with a severity)');
  } finally { rmSync(dir, { recursive: true, force: true }); rmSync(fake, { recursive: true, force: true }); }
});

test('natlang check --idiom without a reviewer says where to point it', () => {
  const dir = program();
  try {
    const ran = natlang(['check', dir, '--idiom'], { NATLANG_IDIOM_REVIEWER: join(dir, 'missing') });
    assert.equal(ran.status, 3);
    assert.match(ran.stderr, /set NATLANG_IDIOM_REVIEWER to the applications\/idiom-reviewer directory/);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('natlang check without --idiom never runs the reviewer (no model)', () => {
  const dir = program();
  try {
    const ran = natlang(['check', dir, '--json'], { NATLANG_IDIOM_REVIEWER: join(dir, 'missing') });
    assert.equal(ran.status, 0, ran.stderr);
    assert.ok(!JSON.parse(ran.stdout).diagnostics.some(item => String(item.code).startsWith('idiom-review-')));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
