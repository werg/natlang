/**
 * Wiring of the coding tools through the natlang runtime: each tool's natural-language function runs with a scripted
 * interpreter whose eval code carries out its numbered steps literally, over the real env service, the crisp helpers
 * and a Node execution environment. Results (or error messages), files, output and diagnostics must equal those of
 * pi-durable's own tools on the same inputs. Wiring evidence only: it says nothing about how a model follows the
 * instructions. Run: node --test applications/pi/test/tools-wiring.test.mjs
 */
import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { cpSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';

const APP = fileURLToPath(new URL('..', import.meta.url));
const HOST = fileURLToPath(new URL('../../../ts-host/dist/index.js', import.meta.url));
const OUT = join(APP, '.natlang', 'test-build-tools');
let natlang, ours, piTools, NodeExecutionEnv;

/** What a small model would write for each function, step by step. */
const SCRIPTS = [
  ['Read the text file args.path for the model', `
    const p = await env.readPath(args.path); if (p.error) throw new Error(p.error.message);
    const r = await env.openReader(p.value); if (r.error) throw new Error(r.error.message);
    let result;
    try {
      for (let attempt = 1; attempt <= 2; attempt++) {
        const before = await env.readerInfo(r.value); if (before.error) throw new Error(before.error.message);
        result = await selection(r.value, before.value, args.path, args.offset, args.limit);
        const after = await env.readerInfo(r.value); if (after.error) throw new Error(after.error.message);
        if (after.value.size > before.value.size || (after.value.size === before.value.size && after.value.mtimeMs === before.value.mtimeMs)) break;
        if (attempt === 2) throw new Error(args.path + ' changed while it was read');
      }
    } finally { await env.closeReader(r.value); }
    return result;`],
  ['Write args.content to the file args.path', `
    const p = await env.toolPath(args.path); if (p.error) throw new Error(p.error.message);
    const held = await env.lock(p.value); if (held.error) throw new Error(held.error.message);
    const w = await env.writeFile(p.value, args.content);
    env.unlock(held.value);
    if (w.error) throw new Error(w.error.message);
    return { content: [{ type: 'text', text: 'Successfully wrote to ' + args.path }] };`],
  ['Apply args.edits to the file args.path', `
    if (!Array.isArray(args.edits) || args.edits.length === 0) throw new Error('Edit tool input is invalid. edits must contain at least one replacement.');
    const p = await env.toolPath(args.path); if (p.error) throw new Error(p.error.message);
    const held = await env.lock(p.value); if (held.error) throw new Error(held.error.message);
    const access = e => new Error('Could not edit file: ' + args.path + '. Error code: ' + e.code + '.');
    try {
      const info = await env.fileInfo(p.value); if (info.error) throw access(info.error);
      if (info.value.kind !== 'file' && info.value.kind !== 'symlink') throw new Error('Could not edit file: ' + args.path + '. Path is not a file.');
      const original = await env.readTextFile(p.value); if (original.error) throw access(original.error);
      const { bom, text: withoutBom } = text.stripBom(original.value);
      const ending = text.detectLineEnding(withoutBom);
      const lf = text.normalizeToLF(withoutBom);
      const applied = await applyEdits(lf, args.edits, args.path);
      if (applied.error) throw new Error(applied.error);
      const written = await env.writeFile(p.value, bom + text.restoreLineEndings(applied.content, ending));
      if (written.error) throw access(written.error);
      env.unlock(held.value);
      const details = env.renderDiff(args.path, applied.base, applied.content);
      return { content: [{ type: 'text', text: 'Successfully replaced ' + args.edits.length + ' block(s) in ' + args.path + '.' }], details };
    } finally { env.unlock(held.value); }`],
  ["content is the file's text with LF line endings", `
    const n = edits.length;
    const es = edits.map(e => ({ oldText: text.normalizeToLF(e.oldText), newText: text.normalizeToLF(e.newText) }));
    for (let i = 0; i < n; i++) if (es[i].oldText === '') return { error: n === 1 ? 'oldText must not be empty in ' + path + '.' : 'edits[' + i + '].oldText must not be empty in ' + path + '.' };
    const fuzzy = es.some(e => text.fuzzyFindText(content, e.oldText).usedFuzzyMatch);
    const base = fuzzy ? text.normalizeForFuzzyMatch(content) : content;
    const matches = [];
    for (let i = 0; i < n; i++) {
      const m = text.fuzzyFindText(base, es[i].oldText);
      if (!m.found) return { error: n === 1 ? 'Could not find the exact text in ' + path + '. The old text must match exactly including all whitespace and newlines.' : 'Could not find edits[' + i + '] in ' + path + '. The oldText must match exactly including all whitespace and newlines.' };
      const k = text.countOccurrences(base, es[i].oldText);
      if (k > 1) return { error: n === 1 ? 'Found ' + k + ' occurrences of the text in ' + path + '. The text must be unique. Please provide more context to make it unique.' : 'Found ' + k + ' occurrences of edits[' + i + '] in ' + path + '. Each oldText must be unique. Please provide more context to make it unique.' };
      matches.push({ editIndex: i, matchIndex: m.index, matchLength: m.matchLength, newText: es[i].newText });
    }
    matches.sort((a, b) => a.matchIndex - b.matchIndex);
    for (let i = 1; i < matches.length; i++) if (matches[i - 1].matchIndex + matches[i - 1].matchLength > matches[i].matchIndex)
      return { error: 'edits[' + matches[i - 1].editIndex + '] and edits[' + matches[i].editIndex + '] overlap in ' + path + '. Merge them into one edit or target disjoint regions.' };
    const next = fuzzy ? text.applyReplacementsPreservingUnchangedLines(content, base, matches) : text.applyReplacements(base, matches);
    if (next === content) return { error: n === 1 ? 'No changes made to ' + path + '. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected.' : 'No changes made to ' + path + '. The replacements produced identical content.' };
    return { base: content, content: next };`],
  ['Run args.command.', `
    if (args.timeout !== undefined) {
      if (!Number.isFinite(args.timeout) || args.timeout <= 0) throw new Error('Invalid timeout: must be a finite number of seconds');
      if (args.timeout > 2147483.647) throw new Error('Invalid timeout: maximum is 2147483.647 seconds');
    }
    const run = await env.runShell(args.command, args.timeout);
    const spill = run.value?.spillPath ?? run.error?.spillPath;
    if (spill) env.diagnostic({ severity: 'info', code: 'full_output', message: 'Full output: ' + spill });
    if (run.error?.code === 'timeout') throw new Error('Command timed out after ' + args.timeout + ' seconds');
    if (run.error?.code === 'aborted') throw new Error('Command aborted');
    if (run.error) throw new Error(run.error.message);
    if (run.value.exitCode !== 0) throw new Error('Command exited with code ' + run.value.exitCode);
    return {};`],
];

function scripted() {
  return async ({ messages }) => {
    const { modelTurnsSoFar } = await import(new URL('native/agent.js', 'file://' + HOST).href);
    const last = messages.at(-1);
    if (modelTurnsSoFar(messages) === 0) {
      const opening = String(messages[1].content);
      const script = SCRIPTS.find(([marker]) => opening.includes(marker));
      if (!script) return { calls: [['return_result', { status: 'failed', reason: 'no script for this function' }]] };
      return { calls: [['eval', { code: script[1] }]] };
    }
    const content = String(last.content);
    if (process.env.WIRING_DEBUG) console.error('TURN', modelTurnsSoFar(messages), last.role, content.slice(0, 300));
    if (modelTurnsSoFar(messages) > 4) return { calls: [['return_result', { status: 'failed', reason: 'scripted model stuck: ' + content.slice(0, 200) }]] };
    if (last.role === 'tool' && (/^Error: /.test(content) || /Already performed before the failure|Nothing else from this eval was kept/.test(content)))
      return { calls: [['return_result', { status: 'failed', reason: content.split('\n')[0].replace(/^Error: /, '') }]] };
    return { text: 'done' };
  };
}

before(async () => {
  const { buildProject, formatDiagnostics, createNatlangRuntime } = await import(HOST);
  const result = buildProject({ project: APP, outDir: OUT, runtimeModule: { url: 'file://' + HOST, path: HOST,
    types: HOST.replace(/\.js$/, '.d.ts'), specifiers: ['@natlang/node'] } });
  if (!result.ok) throw new Error(formatDiagnostics(result.diagnostics));
  natlang = createNatlangRuntime({ model: scripted() });
  ours = Object.fromEntries((await import(join(OUT, 'extensions/coding-tools/index.js'))).codingTools(natlang).tools.map(tool => [tool.name, tool]));
  piTools = Object.fromEntries((await import(join(OUT, 'vendor/durable/src/tools/index.js'))).CodingTools.tools.map(tool => [tool.name, tool]));
  ({ NodeExecutionEnv } = await import(join(OUT, 'vendor/durable/src/env/node.js')));
});

/** Run one tool call with a recording api over a Node environment rooted at cwd. */
async function call(tool, args, cwd) {
  const output = [], diagnostics = [];
  const api = { env: new NodeExecutionEnv({ cwd }), taskId: 1, conversationId: 1, callId: 'c', outputWindow: undefined,
    output: (chunk) => output.push(String(chunk)), diagnostic: d => diagnostics.push(d) };
  try {
    const prepared = tool.prepareArguments ? tool.prepareArguments(structuredClone(args)) : args;
    return { result: await tool.execute(prepared, api, BACKGROUND_CONTEXT), output: output.join(''), diagnostics };
  } catch (error) { return { error: error.message.replace(/pi-(ours|pi)-\w+/g, 'DIR'), output: output.join(''), diagnostics }; }
}

const snapshot = dir => Object.fromEntries(readdirSync(dir, { recursive: true }).sort().map(name => {
  try { return [name, readFileSync(join(dir, name), 'utf8')]; } catch { return [name, null]; }
}));

/** Both implementations on copies of the same directory: equal outcomes and equal files afterwards. */
async function same(name, args, files) {
  const base = mkdtempSync(join(tmpdir(), 'pi-base-'));
  for (const [path, content] of Object.entries(files)) writeFileSync(join(base, path), content);
  const a = mkdtempSync(join(tmpdir(), 'pi-ours-')), b = mkdtempSync(join(tmpdir(), 'pi-pi-'));
  cpSync(base, a, { recursive: true }); cpSync(base, b, { recursive: true });
  const mine = await call(ours[name], args, a), theirs = await call(piTools[name], args, b);
  const scrub = value => JSON.parse(JSON.stringify(value ?? null).replaceAll(a, 'DIR').replaceAll(b, 'DIR').replace(/Full output: [^"]*/g, 'Full output: SPILL'));
  assert.deepEqual(scrub(mine), scrub(theirs), `${name} ${JSON.stringify(args)}`);
  assert.deepEqual(snapshot(a), snapshot(b), `${name} files ${JSON.stringify(args)}`);
  return mine;
}

test('the tool declarations and order are pi\'s', () => {
  assert.deepEqual(Object.keys(ours), ['read', 'write', 'edit', 'bash']);
  for (const name of Object.keys(ours)) {
    assert.equal(ours[name].description, piTools[name].description);
    assert.deepEqual(JSON.parse(JSON.stringify(ours[name].parameters)), JSON.parse(JSON.stringify(piTools[name].parameters)));
    assert.deepEqual(ours[name].outputLimits, piTools[name].outputLimits);
    assert.equal(ours[name].replay, piTools[name].replay);
  }
});

test('read equals pi', async () => {
  const many = Array.from({ length: 2600 }, (_, i) => `line ${i + 1}`).join('\n') + '\n';
  const wide = 'w'.repeat(60 * 1024) + '\nnext\n';
  const files = { 'a.txt': 'one\ntwo\nthree\n', 'many.txt': many, 'wide.txt': wide, 'bom.txt': '﻿first\nsecond', 'empty.txt': '',
    'utf.txt': 'é€😀\n'.repeat(20000), 'img.gif': 'GIF89a\x01\x00\x01\x00' };
  for (const args of [{ path: 'a.txt' }, { path: 'a.txt', offset: 2 }, { path: 'a.txt', offset: 2, limit: 1 }, { path: 'a.txt', offset: 9 },
    { path: 'a.txt', limit: 0 }, { path: 'a.txt', limit: -1 }, { path: 'many.txt' }, { path: 'many.txt', offset: 2500 }, { path: 'many.txt', offset: 10, limit: 5 },
    { path: 'wide.txt' }, { path: 'wide.txt', offset: 2 }, { path: 'bom.txt' }, { path: 'empty.txt' }, { path: 'utf.txt' }, { path: 'img.gif' },
    { path: 'missing.txt' }, { path: '@a.txt' }, { path: '.' }]) await same('read', args, files);
});

test('write equals pi', async () => {
  await same('write', { path: 'new.txt', content: 'hello\n' }, {});
  await same('write', { path: 'a.txt', content: 'over' }, { 'a.txt': 'old' });
  await same('write', { path: 'deep/er/x.txt', content: 'x' }, {});
});

test('edit equals pi', async () => {
  const files = { 'a.txt': 'alpha\nbeta\ngamma\nbeta2\n', 'crlf.txt': '﻿one\r\ntwo\r\nthree\r\n', 'q.txt': 'say ’hi’  \nend\n' };
  for (const args of [
    { path: 'a.txt', edits: [{ oldText: 'gamma', newText: 'GAMMA' }] },
    { path: 'a.txt', edits: [{ oldText: 'alpha', newText: 'A' }, { oldText: 'gamma', newText: 'G' }] },
    { path: 'a.txt', edits: [{ oldText: 'beta', newText: 'B' }] },
    { path: 'a.txt', edits: [{ oldText: 'nope', newText: 'B' }] },
    { path: 'a.txt', edits: [{ oldText: 'alpha\nbeta', newText: 'x' }, { oldText: 'beta\ngamma', newText: 'y' }] },
    { path: 'a.txt', edits: [{ oldText: '', newText: 'x' }] },
    { path: 'a.txt', edits: [{ oldText: 'gamma', newText: 'gamma' }] },
    { path: 'a.txt', edits: [] },
    { path: 'a.txt', edits: '[{"oldText":"gamma","newText":"G"}]' },
    { path: 'a.txt', oldText: 'gamma', newText: 'G2' },
    { path: 'crlf.txt', edits: [{ oldText: 'two\nthree', newText: '2\n3' }] },
    { path: 'q.txt', edits: [{ oldText: "say 'hi'", newText: 'said' }] },
    { path: 'missing.txt', edits: [{ oldText: 'a', newText: 'b' }] },
    { path: '.', edits: [{ oldText: 'a', newText: 'b' }] },
  ]) await same('edit', args, files);
});

test('bash equals pi', async () => {
  for (const args of [{ command: 'echo hello; echo err >&2' }, { command: 'exit 3' }, { command: 'echo x', timeout: 0 },
    { command: 'echo x', timeout: 1e9 }, { command: 'sleep 5', timeout: 0.3 }, { command: 'seq 1 5000' }, { command: 'pwd' }]) {
    const outcome = await same('bash', args, {});
    if (args.command === 'seq 1 5000') assert.match(outcome.diagnostics[0]?.message ?? '', /^Full output: /);
  }
});
