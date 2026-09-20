import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dumpNativeState, loadFunctionFile, NatlangHost } from '../dist/index.js';
import { BuildWorkspace } from '../../applications/build_workbench.mjs';
import { MediaWorkspace } from '../../applications/media_workbench.mjs';
import { RecipeTerminal } from '../../applications/semantic_terminal.mjs';

const stepPath = fileURLToPath(new URL('../../codebases/semantic_terminal/step.nl', import.meta.url));
const request = (id, text) => ({ kind: 'request', id, request_id: '', job_id: '', text,
  status: '', detail: '' });

test('natlang stream terminal routes real build and media recipes with correlated completions', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-terminal-'));
  writeFileSync(join(folder, 'source.txt'), 'hello');
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin',
    '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=8', '-t', '1', '-threads', '1',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', join(folder, 'input.mp4')]);
  const build = await new BuildWorkspace(folder).open();
  const media = await new MediaWorkspace(folder).open();
  const terminal = new RecipeTerminal([
    { id: 'compile-note', description: 'build a text artifact', run: async () => {
      const result = await build.execute({ id: 'note', argv: [process.execPath, '-e',
        'require("fs").copyFileSync("source.txt", "note.txt")'],
      inputs: ['source.txt'], outputs: ['note.txt'] });
      return { status: result.status, detail: result.output_sha256 || result.detail };
    } },
    { id: 'scale-video', description: 'scale a video', run: async () => {
      const result = await media.render({ kind: 'scale', input: 'input.mp4', output: 'scaled.mp4',
        start: 0, end: 0, x: 0, y: 0, width: 80, height: 60, keep_audio: false });
      return { status: result.status, detail: result.sha256 || result.detail };
    } },
  ]);
  const native = { terminal, drainEvents: () => [
    ...terminal.drainEvents(), ...build.drainEvents(), ...media.drainEvents() ] };
  const host = new NatlangHost({ host: native, mode: 'retained' });
  const step = dumpNativeState(loadFunctionFile(stepPath));
  const init = { revision: 0, active_request: '', active_job: '', status: 'idle', messages: [], history: [] };
  const tracePath = join(folder, 'terminal.trace.jsonl');
  async function* events() {
    yield request('r1', 'Build the note.');
    yield await terminal.wait('r1');
    yield request('r2', 'Scale the video.');
    yield await terminal.wait('r2');
  }
  let stepCount = 0, interpretation = 0;
  try {
    const result = await host.run({ source: { kind: 'program', program: { $fold: {
      type: 'Fold<Event, Session>', types: step.$lambda.types, init, step } } },
      streams: { over: events() }, tracePath, options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function step(')) {
          const index = stepCount++;
          if (index % 2 === 0) return { calls: [
            ['call', { function: 'recipes', to: 'let/catalog' }],
            ['call', { function: 'interpret', to: 'let/recipe', inputs: {
              text: 'args/item/text', catalog: 'let/catalog' } }],
            ['call', { function: 'launch', to: 'return', inputs: {
              acc: 'args/acc', item: 'args/item', recipe: 'let/recipe' } }],
          ], completion_tokens: 1 };
          return { calls: [
            ['call', { function: 'explain', to: 'let/message', inputs: { item: 'args/item' } }],
            ['call', { function: 'settle', to: 'return', inputs: {
              acc: 'args/acc', item: 'args/item', message: 'let/message' } }],
          ], completion_tokens: 1 };
        }
        if (prompt.includes('Choose one recipe ID')) return { calls: [['write', {
          path: 'return', value: interpretation++ === 0 ? 'compile-note' : 'scale-video' }]],
          completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: 'Completed with recorded outcome.' }]],
          completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'ok');
    assert.equal(result.value.revision, 4);
    assert.deepEqual(Array.from(result.value.history).map(row => row.request_id), ['r1', 'r2']);
    assert.equal(readFileSync(join(folder, 'note.txt'), 'utf8'), 'hello');
    assert.ok(readFileSync(join(folder, 'scaled.mp4')).length > 1000);
    const trace = readFileSync(tracePath, 'utf8').trim().split('\n').map(JSON.parse);
    assert.equal(trace.filter(e => e.kind === 'host' && e.event?.operation === 'terminal.complete').length, 2);
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('cancellation preserves the actual result and forged completion cannot settle a session', async () => {
  const terminal = new RecipeTerminal([{ id: 'work', description: 'finish work',
    run: async () => ({ status: 'ok', detail: 'finished' }) }]);
  const job = terminal.start('r1', 'work');
  terminal.cancel(job.id);
  const event = await terminal.wait('r1');
  assert.equal(event.status, 'ok');
  assert.match(event.detail, /Cancellation was requested/);
  const host = new NatlangHost({ host: { terminal } });
  const step = dumpNativeState(loadFunctionFile(stepPath));
  const definition = step.$lambda.codebase.settle;
  const source = { kind: 'program', program: { $lambda: {
    type: 'Lambda<{ acc: Session, item: Event, message: Text }, Session>',
    types: step.$lambda.types, engine: definition.engine, code: definition.code } } };
  const acc = { revision: 2, active_request: 'r1', active_job: job.id,
    status: 'cancel-requested', messages: [], history: [] };
  try {
    const forged = await host.run({ source,
      inputs: { acc, item: { ...event, status: 'failed' }, message: 'forged failure' } });
    assert.equal(forged.value.status, 'cancel-requested');
    assert.equal(forged.value.history.length, 0);
    const real = await host.run({ source,
      inputs: { acc, item: event, message: 'Work completed despite cancellation request.' } });
    assert.equal(real.value.status, 'ok');
    assert.equal(real.value.history[0].status, 'ok');
  } finally { host.close(); }
});
