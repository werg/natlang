/**
 * The stable prefix every engine's prompt cache and the learner's prefix tree reuse (plans/semantic-fp/02 R0). LFM3's
 * chat template writes the system prompt, then the tool list, into the system turn: a call shares the cached prefix of
 * another only up to the first byte that differs. So the system message is one text for every tool-loop call, and the
 * tool list opens with the tools every call has (STABLE_TOOLS); the tools a run or a call shape adds follow, and
 * return_result (its value typed by the call's declared result) is last. Everything else per call goes after the call's
 * instructions in its opening message (callNotes).
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { cpSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime } from '../dist/index.js';
import { STABLE_TOOLS } from '../dist/native/agent.js';
import { SYSTEM_PROMPT } from '../dist/native/prompt.js';
import { DECISION_SYSTEM_PROMPT } from '../dist/native/decision.js';
import { servingTools } from '../dist/teacher/serving-request.js';
import { parseType, TypeEnv } from '../dist/native/types.js';
import { scriptedModel } from './support/natlang.mjs';

const name = tool => (tool.function ?? tool).name;
const line = tool => JSON.stringify(tool);

/** Every call shape the runtime has: plain, own code, folder reducer (read and write, each file-tool surface), depth
 * limit, and declared results of different types (Neuralese and finite ones included). */
function shapes() {
  const results = ['string', 'boolean', "'low' | 'high'", '{ answer: string; score: number }', 'Neuralese<string>', 'string[]'];
  const out = [];
  for (const returns of results)
    for (const ownCode of [false, true])
      for (const folder of [null, 'read', 'write'])
        for (const fileTools of folder ? ['all', 'editor', 'files'] : ['all'])
          for (const depthLimited of [false, true])
            out.push({ label: JSON.stringify({ returns, ownCode, folder, fileTools, depthLimited }),
              tools: servingTools({ ownCode, folder, fileTools, depthLimited }, { returns: parseType(returns), env: new TypeEnv() }) });
  return out;
}

test('the tool list opens with the same tools for every call shape, and return_result is last', () => {
  const all = shapes();
  const reference = all[0].tools.slice(0, STABLE_TOOLS.length).map(line);
  assert.deepEqual(all[0].tools.slice(0, STABLE_TOOLS.length).map(name), [...STABLE_TOOLS]);
  for (const shape of all) {
    assert.deepEqual(shape.tools.slice(0, STABLE_TOOLS.length).map(line), reference, `stable head differs for ${shape.label}`);
    assert.equal(name(shape.tools.at(-1)), 'return_result', shape.label);
    // Nothing a call shape adds precedes the stable head; the code-editing tools precede the file tools.
    const names = shape.tools.map(name);
    assert.ok(!names.slice(STABLE_TOOLS.length).some(tool => STABLE_TOOLS.includes(tool)), shape.label);
    const edit = names.indexOf('edit_code'), firstFile = names.findIndex(tool => /_files?$|^bash$|^python$|^editor$|^delegate$/.test(tool));
    if (edit >= 0 && firstFile >= 0) assert.ok(edit < firstFile, shape.label);
  }
  // Same shape, different declared result: only the last line differs.
  const plain = all.filter(shape => shape.label.includes('"ownCode":false,"folder":null') && shape.label.includes('"depthLimited":false'));
  assert.ok(plain.length > 3);
  for (const shape of plain) assert.deepEqual(shape.tools.slice(0, -1).map(line), plain[0].tools.slice(0, -1).map(line));
  assert.ok(new Set(plain.map(shape => line(shape.tools.at(-1)))).size > 1, 'return_result carries the declared result');
});

test('a real run: every tool-loop request sends one system message, its tools open with the stable head, and decision readouts share one system message', async () => {
  const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
    types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };
  const outDir = mkdtempSync(join(tmpdir(), 'natlang-prefix-'));
  // examples/triage scores classify and is_urgent as decision readouts (B4); without `readout:` they run the tool loop
  // with three different declared results (Label, boolean, string), which is what this test compares.
  const project = mkdtempSync(join(tmpdir(), 'natlang-prefix-triage-'));
  cpSync(fileURLToPath(new URL('../../examples/triage', import.meta.url)), project, { recursive: true });
  for (const file of readdirSync(project).filter(name => name.endsWith('.nl')))
    writeFileSync(join(project, file), readFileSync(join(project, file), 'utf8').replace(/^readout: decision\n/m, ''));
  const built = buildProject({ project, outDir, runtimeModule, writeDeclarations: false });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const { triage } = await import(pathToFileURL(join(outDir, 'triage.js')).href);
  const scripted = scriptedModel(opening => {
    if (opening.includes('Pick the label')) return 'return "billing"';
    if (opening.includes('A ticket is urgent') || opening.includes('attention within the hour')) return 'return true';
    if (opening.includes('Write one paragraph')) return 'return "Checkout is down."';
    if (opening.includes('noticeably shorter')) return 'return "Checkout down."';
    return 'return true';   // property checks (holds) and any other judgment
  });
  const requests = [];
  const driver = async request => {
    requests.push(structuredClone({ messages: request.messages, tools: request.tools ?? [] }));
    if (!request.tools?.length) {   // a decision readout: answer with one of its offered values
      const offered = /these JSON values and nothing else: (.*)\.$/.exec(String(request.messages.at(-1)?.content ?? ''))?.[1];
      return { text: JSON.stringify(offered ? JSON.parse(`[${offered}]`)[0] : true), calls: [] };
    }
    return scripted.driver(request);
  };
  const tickets = JSON.parse(readFileSync(join(project, 'tickets.json'), 'utf8'));
  await createNatlangRuntime({ model: driver }).run(() => triage(tickets, 'billing, technical, spam'));
  const loops = requests.filter(request => request.tools.length);
  assert.ok(loops.length >= 4, 'tool-loop calls ran');
  for (const request of loops) {
    assert.equal(request.messages[0].role, 'system');
    assert.equal(request.messages[0].content, SYSTEM_PROMPT);
    assert.deepEqual(request.tools.slice(0, STABLE_TOOLS.length).map(line), loops[0].tools.slice(0, STABLE_TOOLS.length).map(line));
    assert.equal(name(request.tools.at(-1)), 'return_result');
  }
  // Different declared results (Label, boolean, string) part only at return_result.
  assert.ok(new Set(loops.map(request => line(request.tools.at(-1)))).size > 1);
  assert.equal(new Set(loops.map(request => request.tools.slice(0, -1).map(line).join('\n'))).size, 1);
  for (const request of requests.filter(request => !request.tools.length))
    if (request.messages[0]?.role === 'system') assert.equal(request.messages[0].content, DECISION_SYSTEM_PROMPT);
});

test('NATLANG_PROMPT_REVISION=2026-11-22 restores the previous texts and tool order byte for byte', () => {
  // sha256 of each text at 6d201f483 (before the 2026-11-23 sentence pass).
  const previous = {
    SYSTEM_PROMPT: '1a012559f79faf7d7a55c918bcca3ea2a06d4c2e10fe6e7f84b0e84da08ff8a5',
    TOOLS_PROMPT_AT_NL_DEPTH_LIMIT: '3890f7d7e5f8910147766adbb332714eb1e9bce9730e36951f8ff9e61914dfd0',
    FUNCTION_TOOLS_PROMPT: '29aeafdb6a7b16680c8cf686426f6c8dad24ae2a66999c965172b6b9b023ae7c',
    GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT: '5b25f398e4bb199afffc4427cc1b03726f8d6f95b4ecef98267aa85e23e2b3a5',
    'directoryReducerPrompt(all,true,true)': 'fb35ee1918755e6c76686b9f7aec84fdc131f9bce3f1d8b1cc5732b1f924af8f',
    'directoryReducerPrompt(editor,false,false)': '202255d7f3baa4aacdf346534551ef416634a7d667eaf8b2aa6f65d82bba262c',
    'directoryReducerPrompt(files,true,true)': '94d12bc10bd2ae53c3bb21e85f1fbbd9e855be2a4a45385e51293c698721f3b9',
    promptAtNlDepthLimit_SYSTEM: 'f22c3ce7c74548ec84d90e8828b730599221cf0778a718a40e19ad9e80b19e60',
  };
  const script = `
    import { createHash } from 'node:crypto';
    const m = await import(${JSON.stringify(new URL('../dist/native/prompt.js', import.meta.url).href)});
    const { servingTools } = await import(${JSON.stringify(new URL('../dist/teacher/serving-request.js', import.meta.url).href)});
    const h = t => createHash('sha256').update(t).digest('hex');
    const out = { SYSTEM_PROMPT: h(m.SYSTEM_PROMPT), TOOLS_PROMPT_AT_NL_DEPTH_LIMIT: h(m.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT),
      FUNCTION_TOOLS_PROMPT: h(m.FUNCTION_TOOLS_PROMPT), GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT: h(m.GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT),
      'directoryReducerPrompt(all,true,true)': h(m.directoryReducerPrompt('all', true, true)),
      'directoryReducerPrompt(editor,false,false)': h(m.directoryReducerPrompt('editor', false, false)),
      'directoryReducerPrompt(files,true,true)': h(m.directoryReducerPrompt('files', true, true)),
      promptAtNlDepthLimit_SYSTEM: h(m.promptAtNlDepthLimit(m.SYSTEM_PROMPT)) };
    const tools = servingTools({ ownCode: true, folder: 'write', fileTools: 'files', depthLimited: false }).map(t => t.function.name);
    console.log(JSON.stringify({ out, tools }));`;
  const run = env => JSON.parse(execFileSync(process.execPath, ['--input-type=module', '-e', script],
    { env: { ...process.env, NATLANG_PROMPT_REVISION: '', NATLANG_MODEL_FACING: '', ...env }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }));
  const old = run({ NATLANG_PROMPT_REVISION: '2026-11-22' });
  assert.deepEqual(old.out, previous);
  assert.deepEqual(old.tools, ['eval', 'read_page', 'list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files',
    'python', 'delegate', 'read_code', 'edit_code', 'diff_code', 'compact_history', 'return_result']);
  // The same opt-out through the series' NATLANG_MODEL_FACING list (held_eval.series() passes it that way).
  assert.deepEqual(run({ NATLANG_MODEL_FACING: 'compact-callable-scope,effect-receipt,prompt-revision-2026-11-22' }).out, previous);
  const current = run({});
  assert.notEqual(current.out.SYSTEM_PROMPT, previous.SYSTEM_PROMPT);
  assert.deepEqual(current.tools, ['eval', 'read_page', 'read_code', 'compact_history', 'edit_code', 'diff_code', 'list_files', 'search_files',
    'read_file', 'write_file', 'edit_file', 'diff_files', 'python', 'delegate', 'return_result']);
  assert.throws(() => run({ NATLANG_PROMPT_REVISION: '2026-01-01' }));
});
