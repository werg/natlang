import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost, NodeFileTree } from '../dist/index.js';
import { MediaWorkspace } from '../../applications/media_workbench.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const source = fileURLToPath(new URL('../../codebases/media_workbench/transform.nl', import.meta.url));

function fixture() {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-media-'));
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin',
    '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=10',
    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100',
    '-t', '2', '-shortest', '-threads', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', join(folder, 'input.mp4')]);
  return folder;
}

async function transform(folder, plan, { vision = null, assessment = null } = {}) {
  const media = await new MediaWorkspace(folder, { vision }).open();
  const host = new NatlangHost({ host: { media, drainEvents: () => media.drainEvents() }, mode: 'retained' });
  const tracePath = join(folder, 'transform.trace.jsonl');
  try {
    const result = await host.run({ source: { kind: 'file', path: source }, tracePath,
      validationFeedback: 'caller',
      inputs: { request: { text: `Please ${plan.kind} the video`, input: 'input.mp4', output: plan.output }, files: new NodeFileTree(folder) },
      modelTurn: request => {
        const prompt = String(request.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function transform(')) return evalTurn(request,
          'const source = await probe(request.input); const chosen = await choose(request, source, files); const receipt = await render(request, source, chosen); const inspection = await inspect(request, chosen, receipt); const assessment = await assess(request, source, chosen, receipt, inspection, files); await finalize(request, source, chosen, receipt, inspection, assessment)');
        if (prompt.includes('Choose exactly one')) return evalTurn(request, `(${JSON.stringify(plan)})`);
        return evalTurn(request, `(${JSON.stringify(assessment ?? {
          intent_met: true, needs_visual_review: false, explanation: 'The transform matches the request.'
        })})`);
      } });
    const trace = readFileSync(tracePath, 'utf8').trim().split('\n').map(JSON.parse);
    return { result, trace };
  } finally { host.close(); }
}

const base = { input: 'input.mp4', output: 'output.mp4', start: 0, end: 0,
  x: 0, y: 0, width: 0, height: 0, keep_audio: true };

test('natlang chooses and verifies a real trimmed clip', async () => {
  const folder = fixture();
  try {
    const { result, trace } = await transform(folder, { ...base, kind: 'trim', start: 0.4, end: 1.4 });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'verified');
    assert.equal(result.value.inspection.width, 320);
    assert.equal(result.value.inspection.height, 240);
    assert.equal(result.value.inspection.has_audio, true);
    assert.ok(Math.abs(result.value.inspection.duration - 1) < 0.16);
    assert.equal(trace.filter(e => e.kind === 'host' && e.event?.operation === 'media.render' && e.event.status === 'ok').length, 1);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('crop requires visual review when no inspector is supplied', async () => {
  const folder = fixture();
  try {
    const { result } = await transform(folder, { ...base, kind: 'crop', x: 0, y: 0,
      width: 160, height: 120 });
    assert.equal(result.value.technical_ok, true);
    assert.equal(result.value.status, 'review');
    assert.equal(result.value.needs_visual_review, true);
    assert.equal(result.value.inspection.visual_status, 'unavailable');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a crop uses the optional visual inspector and records its model identity', async () => {
  const folder = fixture();
  let sampled = false;
  try {
    const { result, trace } = await transform(folder, { ...base, kind: 'crop',
      x: 0, y: 0, width: 160, height: 120 }, { vision: async ({ frame }) => {
      assert.ok(readFileSync(frame).length > 100);
      sampled = true;
      return { status: 'supported', detail: 'subject remains visible', model_id: 'fixture-inspector' };
    } });
    assert.equal(result.value.status, 'verified');
    assert.equal(sampled, true);
    assert.ok(trace.some(e => e.kind === 'host' && e.event?.operation === 'media.vision' &&
      e.event.model_id === 'fixture-inspector'));
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('an impossible crop and corrupt input cannot report success', async () => {
  const folder = fixture();
  try {
    const { result } = await transform(folder, { ...base, kind: 'crop', x: 300, y: 0,
      width: 160, height: 120 });
    assert.equal(result.value.status, 'failed');
    assert.equal(result.value.receipt.status, 'failed');
    assert.match(result.value.receipt.detail, /outside source frame/);
    writeFileSync(join(folder, 'corrupt.mp4'), 'not a video');
    const media = await new MediaWorkspace(folder).open();
    assert.equal((await media.probe('corrupt.mp4')).status, 'failed');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
