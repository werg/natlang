import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, checkProject, createNatlangRuntime, compileVirtualProject } from '../dist/index.js';
import * as runtimeNamespace from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const RUNTIME = { url: new URL('../dist/index.js', import.meta.url).href,
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

function project(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-project-'));
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

const APP = {
  'package.json': JSON.stringify({ name: 'fixture-app', private: true, type: 'module' }),
  'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext',
    strict: true, skipLibCheck: true, rootDir: 'src', outDir: 'dist' }, include: ['src/**/*.ts'] }),
  'src/types.ts': 'export type Ticket = { id: string, text: string };\n',
  'src/triage.nl': '---\nargs:\n  ticket: Ticket\nreturns: \'"urgent" | "normal"\'\n---\nDecide whether ticket is urgent.\n',
  'src/triage/keywords.ts': 'export const urgent = ["outage", "down"];\n',
  'src/natlang.d/summarize.nl': '---\nargs:\n  text: string\nreturns: string\n---\nSummarize text in three words.\n',
  'src/natlang.d/tone.ts': 'export function polite(text: string): string { return `Please: ${text}`; }\n',
  'src/app.ts': `import { nl, iterateOn, createNatlangRuntime } from '@natlang/node';
import triage from './triage.nl';
import type { Ticket } from './types.js';

export type Report = { id: string, priority: 'urgent' | 'normal', note: string };

export async function handle(ticket: Ticket, style: string): Promise<Report> {
  const priority = await triage(ticket);
  let seen = 0;
  const note: string = await nl\`Write a note for ticket in style; count it in seen.\`(ticket);
  const improved = await iterateOn(async (draft: string) => draft + '!', note).until(draft => draft.endsWith('!!'));
  return { id: ticket.id, priority, note: improved + ' seen=' + seen };
}

export async function main(tickets: Ticket[], style: string): Promise<Report[]> {
  return Promise.all(tickets.map(ticket => handle(ticket, style)));
}
`,
};

test('natlang build types .nl imports, plans inline lambdas, embeds records, and runs against this runtime', async () => {
  const root = project(APP);
  const result = buildProject({ project: root, runtimeModule: RUNTIME });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics, null, 1));
  const declaration = readFileSync(join(root, 'src/triage.d.nl.ts'), 'utf8');
  assert.match(declaration, /declare const fn: NatlangFunction<\[ticket: Ticket\], "urgent" \| "normal"> & \{ readonly keywords: \{ readonly urgent: string\[\] \} \}/);
  assert.equal(result.manifest.inline.length, 1);
  assert.equal(result.manifest.inline[0].signature, '(ticket: Ticket) => string');
  assert.deepEqual(result.manifest.named.map(item => item.source), ['src/triage.nl']);
  const emitted = readFileSync(join(root, 'dist/app.js'), 'utf8');
  assert.match(emitted, /from "\.\/triage\.nl\.js"/);
  assert.match(readFileSync(join(root, 'dist/triage.nl.js'), 'utf8'), /__natlang\.named\("triage"/);
  assert.match(emitted, /\bnl\.__inline\(/);
  assert.match(emitted, /withCompilerSite\("src\/app\.ts#improved"\)/);
  assert.ok(emitted.includes(RUNTIME.url), 'the emitted runtime import is bound to the running runtime');

  const model = scriptedModel(opening => {
    if (opening.includes('Decide whether ticket is urgent')) return 'return keywords.urgent.some(word => ticket.text.includes(word)) ? "urgent" : "normal"';
    if (opening.includes('Write a note')) return 'seen = seen + 1; return tone.polite(style + " " + ticket.id)';
    return null;
  });
  const app = await import(pathToFileURL(join(root, 'dist/app.js')).href);
  const runtime = createNatlangRuntime({ model: model.driver });
  const reports = await runtime.run(() => app.main([{ id: 'T1', text: 'site is down' }, { id: 'T2', text: 'typo' }], 'brief'));
  assert.deepEqual(reports, [
    { id: 'T1', priority: 'urgent', note: 'Please: brief T1!! seen=1' },
    { id: 'T2', priority: 'normal', note: 'Please: brief T2!! seen=1' }]);
  const inlineOpening = model.openings.find(opening => opening.includes('Write a note'));
  assert.match(inlineOpening, /let style: string = "brief"; \/\/ assignments are written back to the caller/);
  assert.match(inlineOpening, /declare function summarize\(text: string\): Promise<string>; {2}\/\/ natural language/, 'natlang.d is the inline callable context');
  assert.match(inlineOpening, /declare namespace tone \{/);
});

test('natlang check reports inline type errors and callable-folder policy violations with locations', () => {
  const root = project({ ...APP,
    'src/bad.ts': "import { nl } from '@natlang/node';\nexport async function f(x: string) { const y = await nl`Guess x`(x); return y; }\n",
    'src/natlang.d/spin.ts': 'export function spin(): number { while (true) {} }\n' });
  const result = checkProject(root, { runtimeModule: RUNTIME });
  assert.equal(result.ok, false);
  const codes = result.diagnostics.map(item => `${item.file}:${item.code}`);
  assert.ok(codes.includes('src/bad.ts:nl-unknown-return'), codes.join('\n'));
  assert.ok(result.diagnostics.some(item => item.file.endsWith('natlang.d') && /while/.test(item.message)), codes.join('\n'));
  assert.equal(existsSync(join(root, 'dist/app.js')), false, 'check does not emit JavaScript');
});

test('a virtual project compiles and runs in-process through the same compiler', async () => {
  const model = scriptedModel(opening => opening.includes('Name a color') ? 'return "teal"' : null);
  const compiled = compileVirtualProject({ files: {
    'main.ts': "import { nl } from '@natlang/browser';\nexport async function main(input: { mood: string }): Promise<string> {\n" +
      '  const color: string = await nl`Name a color for the mood in input.`(input);\n  return color.toUpperCase();\n}\n' } }, runtimeNamespace, { target: 'node' });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  const runtime = createNatlangRuntime({ model: model.driver });
  assert.equal(await runtime.run(() => compiled.require('main.ts').main({ mood: 'calm' })), 'TEAL');
});
