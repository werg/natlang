import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { MediaWorkspace, assemblePlan, checkPlan, containerOf, transform as runTransform } from '../../applications/dist/media/index.js';
import { scriptedModel } from './support/natlang.mjs';

// The workbench drives the real ffmpeg; without it these tests cannot run.
const missingFfmpeg = spawnSync('ffmpeg', ['-version']).status === 0 ? false : 'ffmpeg is not installed';

function fixture() {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-media-'));
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-nostdin',
    '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=10',
    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=44100',
    '-t', '2', '-shortest', '-threads', '1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-c:a', 'aac', join(folder, 'input.mp4')]);
  return folder;
}

const STAGES = [['operation', 'Decide which one operation'], ['trim', 'Work out the span of the clip'], ['crop', 'Work out the rectangle'],
  ['scale', 'Work out the target size'], ['transcode', 'Work out the container'], ['audio', 'Decide whether the output keeps the sound'],
  ['picture', 'Decide whether the request depends'], ['assess', 'Judge whether the rendered clip'], ['note', 'Find the supporting file']];
const stageOf = opening => STAGES.find(([, text]) => opening.includes(text))?.[0];
const OK = { intent_met: true, explanation: 'The transform matches the request.' };

/**
 * Run `transform` with scripted stages. `answers` maps a stage to its result, or to a function of the 1-based call number;
 * `calls` lists the stages in the order the model was asked.
 */
async function transform(folder, answers, { vision = null, text = null, output = 'output.mp4', files = true } = {}) {
  const media = await new MediaWorkspace(folder, { vision }).open();
  const calls = [];
  const model = scriptedModel(opening => {
    const stage = stageOf(opening);
    calls.push(stage);
    const count = calls.filter(name => name === stage).length;
    const answer = typeof answers[stage] === 'function' ? answers[stage](count, opening) : answers[stage];
    if (stage === 'note' && answer === undefined) return 'return ""';
    if (stage === 'assess' && answer === undefined) return `return ${JSON.stringify(OK)}`;
    if (stage === 'picture' && answer === undefined) return 'return false';
    if (stage === 'audio' && answer === undefined) return 'return true';
    return typeof answer === 'string' && answer.startsWith('code:') ? answer.slice(5) : `return ${JSON.stringify(answer)}`;
  });
  const value = await createNatlangRuntime({ model: model.driver }).run(() => runTransform(media,
    { text: text ?? `Please ${answers.operation} the video`, input: 'input.mp4', output }, files ? openFolder(folder).root() : undefined));
  return { result: { value }, events: media.drainEvents(), calls, openings: model.openings };
}

const MP4 = { container: 'mp4', video_codec: 'h264' };
test('natlang chooses and verifies a real trimmed clip; the host writes the plan', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const { result, events, calls, openings } = await transform(folder, { operation: 'trim', trim: { start: 0.4, end: 1.4 } });
    assert.equal(result.value.status, 'verified');
    assert.equal(result.value.inspection.width, 320);
    assert.equal(result.value.inspection.height, 240);
    assert.equal(result.value.inspection.has_audio, true);
    assert.equal(result.value.inspection.video_codec, 'h264');
    assert.ok(Math.abs(result.value.inspection.duration - 1) < 0.16);
    const render = events.find(e => e.operation === 'media.render' && e.status === 'ok');
    assert.deepEqual([render.input, render.output, render.plan.kind, render.plan.container, render.plan.video_codec, render.plan.x, render.plan.width],
      ['input.mp4', 'output.mp4', 'trim', 'mp4', 'h264', 0, 0]);
    assert.deepEqual([...calls].sort(), ['assess', 'audio', 'note', 'operation', 'picture', 'trim']);
    // The stages read measurements and the request, not file names or hashes.
    for (const opening of openings.filter(text => !text.includes('Find the supporting file')))
      assert.doesNotMatch(opening, /input\.mp4|output\.mp4|[0-9a-f]{64}/);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('crop requires visual review when no inspector is supplied', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const { result } = await transform(folder, { operation: 'crop', crop: { x: 0, y: 0, width: 160, height: 120 } });
    assert.equal(result.value.technical_ok, true);
    assert.equal(result.value.status, 'review');
    assert.equal(result.value.needs_visual_review, true);
    assert.equal(result.value.inspection.visual_status, 'unavailable');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a request that depends on the picture needs review when the frame is uncertain, and is rejected when contradicted', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const trimmed = { operation: 'trim', trim: { start: 0, end: 1 }, picture: true };
    assert.equal((await transform(folder, trimmed, { output: 'a.mp4' })).result.value.status, 'verified');
    const crop = { operation: 'crop', crop: { x: 0, y: 0, width: 160, height: 120 }, picture: true };
    const { result } = await transform(folder, crop, { output: 'b.mp4', vision: async () => ({ status: 'uncertain', detail: 'blurry' }) });
    assert.equal(result.value.status, 'review');
    const contradicted = await transform(folder, crop, { output: 'c.mp4', vision: async () => ({ status: 'contradicted' }) });
    assert.equal(contradicted.result.value.status, 'rejected');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a crop uses the optional visual inspector and records its model identity', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  let sampled = false;
  try {
    const { result, events } = await transform(folder, { operation: 'crop', crop: { x: 0, y: 0, width: 160, height: 120 } },
      { vision: async ({ frame }) => {
        assert.ok(readFileSync(frame).length > 100);
        sampled = true;
        return { status: 'supported', detail: 'subject remains visible', model_id: 'fixture-inspector' };
      } });
    assert.equal(result.value.status, 'verified');
    assert.equal(sampled, true);
    assert.ok(events.some(e => e.operation === 'media.vision' && e.model_id === 'fixture-inspector'));
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a failed plan check returns to the parameter stage once, with the problem', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const repaired = await transform(folder, { operation: 'crop', crop: (count, opening) => {
      if (count === 1) return { x: 300, y: 0, width: 160, height: 120 };
      assert.match(opening, /outside source frame/);
      return { x: 160, y: 0, width: 160, height: 120 };
    } });
    assert.equal(repaired.calls.filter(stage => stage === 'crop').length, 2);
    assert.equal(repaired.result.value.receipt.status, 'ok');
    assert.equal(repaired.result.value.status, 'review');

    const stuck = await transform(folder, { operation: 'crop', crop: { x: 300, y: 0, width: 160, height: 120 } }, { output: 'stuck.mp4' });
    assert.equal(stuck.calls.filter(stage => stage === 'crop').length, 2, 'one retry, not more');
    assert.equal(stuck.result.value.status, 'failed');
    assert.equal(stuck.result.value.receipt.status, 'failed');
    assert.match(stuck.result.value.receipt.detail, /outside source frame/);
    // Without a technical success the semantic stages do not run.
    assert.deepEqual(stuck.calls.filter(stage => stage === 'assess' || stage === 'picture'), []);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('odd scale sizes are returned to the stage as a problem', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const { result, calls } = await transform(folder, { operation: 'scale', scale: count => count === 1 ? { width: 161, height: 121 } : { width: 160, height: 120 } });
    assert.equal(calls.filter(stage => stage === 'scale').length, 2);
    assert.equal(result.value.status, 'verified');
    assert.equal(result.value.inspection.width, 160);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('an unsupported request is not planned or assessed', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    const { result, calls } = await transform(folder, { operation: 'unsupported' }, { text: 'Add a watermark and then reverse it' });
    assert.equal(result.value.status, 'unsupported');
    assert.deepEqual(calls.filter(stage => !['note', 'operation'].includes(stage)), []);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a request names a note; the stages read it as data', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  writeFileSync(join(folder, 'brief.md'), 'KEEP THE OPENING SECOND');
  try {
    const { result, openings } = await transform(folder, { operation: 'trim', trim: { start: 0, end: 1 },
      note: 'code:return await files.file("brief.md").readText()' }, { text: 'Trim the video as described in brief.md' });
    assert.equal(result.value.status, 'verified');
    const planning = openings.find(text => text.includes('Work out the span of the clip'));
    assert.match(planning, /KEEP THE OPENING SECOND/);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

const hasEncoder = name => spawnSync('ffmpeg', ['-hide_banner', '-encoders']).stdout.toString().includes(name);

test('transcode writes the container and codec the plan names', { skip: missingFfmpeg || !hasEncoder('libvpx-vp9') || !hasEncoder('libopus') }, async () => {
  const folder = fixture();
  try {
    const { result, events } = await transform(folder, { operation: 'transcode', transcode: { container: 'webm', video_codec: 'vp9' } },
      { output: 'output.webm', text: 'Transcode to WebM' });
    assert.equal(result.value.status, 'verified', result.value.explanation);
    assert.equal(result.value.inspection.video_codec, 'vp9');
    assert.equal(events.find(e => e.operation === 'media.render' && e.status === 'ok').plan.container, 'webm');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a transcode whose container does not fit the output name returns to the stage', { skip: missingFfmpeg || !hasEncoder('libx265') }, async () => {
  const folder = fixture();
  try {
    const { result, calls } = await transform(folder, { operation: 'transcode', transcode: count =>
      count === 1 ? { container: 'mkv', video_codec: 'h265' } : { container: 'mp4', video_codec: 'h265' } }, { output: 'out.mp4', text: 'Transcode to H.265' });
    assert.equal(calls.filter(stage => stage === 'transcode').length, 2);
    assert.equal(result.value.status, 'verified', result.value.explanation);
    assert.equal(result.value.inspection.video_codec, 'hevc');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('an impossible crop and corrupt input cannot report success', { skip: missingFfmpeg }, async () => {
  const folder = fixture();
  try {
    writeFileSync(join(folder, 'corrupt.mp4'), 'not a video');
    const media = await new MediaWorkspace(folder).open();
    assert.equal((await media.probe('corrupt.mp4')).status, 'failed');
    const receipt = await media.render({ kind: 'crop', input: 'input.mp4', output: 'x.mp4', start: 0, end: 0, x: 300, y: 0, width: 160, height: 120,
      keep_audio: true, ...MP4 });
    assert.equal(receipt.status, 'failed');
    assert.match(receipt.detail, /outside source frame/);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('checkPlan states exact validity rules; assemblePlan fills identifiers and defaults', () => {
  const source = { width: 320, height: 240, duration: 2 };
  const plan = assemblePlan({ text: 'x', input: 'in.mp4', output: 'out.webm' }, 'trim', true, { start: 0, end: 1 });
  assert.deepEqual([plan.input, plan.output, plan.container, plan.video_codec, plan.x, plan.width], ['in.mp4', 'out.webm', 'webm', 'vp9', 0, 0]);
  assert.equal(checkPlan(plan, source), null);
  assert.match(checkPlan({ ...plan, end: 3 }, source).text, /outside source duration/);
  assert.match(checkPlan({ ...plan, container: 'mp4' }, source).text, /must end in \.mp4/);
  assert.equal(checkPlan({ ...plan, container: 'mp4' }, source).stage, 'output');
  assert.match(checkPlan({ ...plan, output: 'out.mov', container: 'mov', video_codec: 'vp9' }, source).text, /mov holds/);
  assert.match(checkPlan({ ...plan, kind: 'scale', width: 101, height: 100 }, source).text, /even/);
  assert.equal(checkPlan({ ...plan, kind: 'unsupported', end: -5 }, source), null);
  assert.equal(containerOf('clip.MKV'), 'mkv');
  assert.equal(containerOf('clip.avi'), undefined);
});
