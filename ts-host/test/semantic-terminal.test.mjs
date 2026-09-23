import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { BuildWorkspace } from '../../applications/dist/build/index.js';
import { MediaWorkspace } from '../../applications/dist/media/index.js';
import { RecipeTerminal, emptyTerminalSession, step } from '../../applications/dist/terminal/index.js';
import { scriptedModel } from './support/natlang.mjs';

const request = (id, text) => ({ kind: 'request', id, request_id: '', job_id: '', text, status: '', detail: '' });

test('the terminal routes real build and media recipes with correlated completions', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-terminal-'));
  writeFileSync(join(folder, 'source.txt'), 'hello');
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'lavfi', '-i', 'testsrc2=size=160x120:rate=8',
    '-t', '1', '-threads', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', join(folder, 'input.mp4')]);
  const build = await new BuildWorkspace(folder).open();
  const media = await new MediaWorkspace(folder).open();
  const terminal = new RecipeTerminal([
    { id: 'compile-note', description: 'build a text artifact', run: async () => {
      const result = await build.execute({ id: 'note', needs: [], description: 'note',
        argv: [process.execPath, '-e', 'require("fs").copyFileSync("source.txt", "note.txt")'], inputs: ['source.txt'], outputs: ['note.txt'] });
      return { status: result.status, detail: result.output_sha256 || result.detail };
    } },
    { id: 'scale-video', description: 'scale a video', run: async () => {
      const result = await media.render({ kind: 'scale', input: 'input.mp4', output: 'scaled.mp4',
        start: 0, end: 0, x: 0, y: 0, width: 80, height: 60, keep_audio: false });
      return { status: result.status, detail: result.sha256 || result.detail };
    } },
  ]);
  const model = scriptedModel(opening => opening.includes('Choose one recipe ID') ?
    'return text.includes("video") ? "scale-video" : "compile-note"' : 'return `Finished with ${item.status}.`');
  const runtime = createNatlangRuntime({ model: model.driver });
  let session = emptyTerminalSession();
  const apply = async event => { session = await runtime.run(() => step(terminal, session, event)); };
  try {
    await apply(request('r1', 'Build the note.'));
    assert.equal(session.status, 'running');
    await apply(request('r0', 'Also build something else.'));
    assert.match(session.messages.at(-1), /was not accepted while r1 is active/);
    await apply(await terminal.wait('r1'));
    await apply(request('r2', 'Scale the video.'));
    await apply(await terminal.wait('r2'));
    assert.equal(session.status, 'ok');
    assert.equal(session.revision, 4);
    assert.deepEqual(session.history.map(row => row.request_id), ['r1', 'r2']);
    assert.equal(session.messages.at(-1), 'Finished with ok.');
    assert.equal(readFileSync(join(folder, 'note.txt'), 'utf8'), 'hello');
    assert.ok(readFileSync(join(folder, 'scaled.mp4')).length > 1000);
    assert.equal(terminal.drainEvents().filter(event => event.operation === 'terminal.complete').length, 2);
  } finally { terminal.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('cancellation preserves the actual result and a forged completion cannot settle a session', async () => {
  const terminal = new RecipeTerminal([{ id: 'work', description: 'finish work', run: async () => ({ status: 'ok', detail: 'finished' }) }]);
  const job = terminal.start('r1', 'work');
  terminal.cancel(job.id);
  const event = await terminal.wait('r1');
  assert.equal(event.status, 'ok');
  assert.match(event.detail, /Cancellation was requested/);
  const model = scriptedModel(() => 'return "Work completed despite the cancellation request."');
  const runtime = createNatlangRuntime({ model: model.driver });
  const session = { revision: 2, active_request: 'r1', active_job: job.id, status: 'cancel-requested', messages: [], history: [] };
  const forged = await runtime.run(() => step(terminal, session, { ...event, status: 'failed' }));
  assert.equal(forged.status, 'cancel-requested');
  assert.equal(forged.history.length, 0);
  assert.equal(model.openings.length, 0, 'a forged completion is rejected before any model call');
  const real = await runtime.run(() => step(terminal, session, event));
  assert.equal(real.status, 'ok');
  assert.equal(real.history[0].status, 'ok');
});
