/**
 * Semantic media workbench. `transform` probes the source, lets natlang choose one trim/crop/scale/
 * transcode plan, renders it with FFmpeg, inspects the output, has natlang assess intent, and
 * finalizes only when exact metadata checks agree. `MediaWorkspace` keeps media bytes, sampled frames
 * and FFmpeg processes on the host; an optional `vision` callback judges a sampled frame for crops.
 */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { lstat, mkdtemp, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { isAbsolute, join, relative, resolve } from 'node:path';
import type { FolderHandle } from '@natlang/node';
import choose from './choose.nl';
import assess from './assess.nl';
import type { Clip, Inspection, MediaRequest, Plan, Receipt } from './types.js';

export type * from './types.js';
export type MediaResult = { status: 'verified' | 'review' | 'rejected' | 'failed' | 'unknown' | 'unsupported', output: string,
  technical_ok: boolean, semantic_ok: boolean, needs_visual_review: boolean, explanation: string, receipt: Receipt, inspection: Inspection };
export type VisionInspector = (input: { frame: string, request: string, plan: Plan, output_sha256: string }) =>
  Promise<{ status: string, detail?: string, model_id?: string }>;

async function sha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

type RunResult = { code: number, signal: string, timedOut: boolean, output: string };
function run(argv: string[], { timeoutMs = 120_000, maxOutput = 8192 } = {}): Promise<RunResult> {
  return new Promise(resolveRun => {
    const child = spawn(argv[0]!, argv.slice(1), { shell: false, stdio: ['ignore', 'pipe', 'pipe'] });
    let output = '', settled = false, timedOut = false;
    const append = (chunk: Buffer) => { if (output.length < maxOutput) output += chunk.toString().slice(0, maxOutput - output.length); };
    child.stdout.on('data', append); child.stderr.on('data', append);
    const timer = setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, timeoutMs);
    const finish = ({ code = -1, signal = '', error = '' }: { code?: number, signal?: string, error?: string }) => {
      if (settled) return;
      settled = true; clearTimeout(timer);
      resolveRun({ code, signal, timedOut, output: error ? `${error}: ${output}` : output });
    };
    child.on('error', error => finish({ error: error.message }));
    child.on('close', (code, signal) => finish({ code: code ?? -1, signal: signal ?? '' }));
  });
}

const outside = (rel: string) => !rel || rel === '..' || rel.startsWith('../') || isAbsolute(rel);

export class MediaWorkspace {
  private readonly rootPath: string;
  private root: string | null = null;
  private readonly ffmpeg: string;
  private readonly ffprobe: string;
  private readonly vision: VisionInspector | null;
  private readonly timeoutMs: number;
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, { ffmpeg = 'ffmpeg', ffprobe = 'ffprobe', vision = null as VisionInspector | null, timeoutMs = 120_000 } = {}) {
    this.rootPath = resolve(root);
    this.ffmpeg = ffmpeg; this.ffprobe = ffprobe; this.vision = vision; this.timeoutMs = timeoutMs;
  }

  async open(): Promise<this> { this.root = await realpath(this.rootPath); return this; }

  private async path(id: string, { output = false } = {}): Promise<string> {
    const root = this.root;
    if (!root || typeof id !== 'string' || !id || isAbsolute(id)) throw new Error(`invalid media ID: ${id}`);
    const target = resolve(root, id);
    if (outside(relative(root, target))) throw new Error(`media ID escapes workspace: ${id}`);
    if (output) {
      const parent = await realpath(resolve(target, '..'));
      if (parent !== root && outside(relative(root, parent))) throw new Error(`output parent escapes workspace: ${id}`);
      try { await lstat(target); throw new Error(`output already exists: ${id}`); }
      catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
      return target;
    }
    const actual = await realpath(target);
    if (outside(relative(root, actual)) || !(await lstat(actual)).isFile()) throw new Error(`input escapes workspace or is not a file: ${id}`);
    return actual;
  }

  async probe(id: string): Promise<Clip> {
    try {
      const path = await this.path(id);
      const result = await run([this.ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', path],
        { timeoutMs: this.timeoutMs, maxOutput: 1_000_000 });
      if (result.code !== 0 || result.timedOut) throw new Error(`ffprobe failed: ${result.output}`);
      const parsed = JSON.parse(result.output) as { streams?: Record<string, unknown>[], format?: { duration?: string } };
      const video = parsed.streams?.find(stream => stream.codec_type === 'video');
      const duration = Number(parsed.format?.duration ?? video?.duration);
      if (!video || !Number.isFinite(duration) || duration <= 0) throw new Error('no video stream or finite duration');
      const clip: Clip = { id, status: 'ok', width: Number(video.width), height: Number(video.height), duration,
        has_audio: parsed.streams!.some(stream => stream.codec_type === 'audio'), sha256: await sha256(path), detail: '' };
      this.events.push({ operation: 'media.probe', id, status: 'ok', sha256: clip.sha256 });
      return clip;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      this.events.push({ operation: 'media.probe', id, status: 'failed', detail });
      return { id, status: 'failed', width: 0, height: 0, duration: 0, has_audio: false, sha256: '', detail };
    }
  }

  async render(plan: Plan): Promise<Receipt> {
    const fail = (status: Receipt['status'], detail: string, exit_code = -1): Receipt => {
      this.events.push({ operation: 'media.render', status, input: plan.input, output: plan.output, detail });
      return { status, output: plan.output, exit_code, sha256: '', detail };
    };
    try {
      const source = await this.probe(plan.input);
      if (source.status !== 'ok') return fail('failed', source.detail);
      const output = await this.path(plan.output, { output: true });
      if (!['trim', 'crop', 'scale', 'transcode'].includes(plan.kind)) return fail('unsupported', `unsupported transform: ${plan.kind}`);
      if (typeof plan.keep_audio !== 'boolean') return fail('failed', 'keep_audio must be boolean');
      if (!([plan.start, plan.end, plan.x, plan.y, plan.width, plan.height]).every(Number.isFinite))
        return fail('failed', 'plan dimensions and times must be finite');
      if (plan.kind === 'trim' && (plan.start < 0 || plan.end <= plan.start || plan.end > source.duration + 0.02))
        return fail('failed', 'trim interval is outside source duration');
      if (plan.kind === 'crop' && (plan.x < 0 || plan.y < 0 || plan.width <= 0 || plan.height <= 0 ||
          plan.x + plan.width > source.width || plan.y + plan.height > source.height ||
          [plan.x, plan.y, plan.width, plan.height].some(n => !Number.isInteger(n))))
        return fail('failed', 'crop rectangle is outside source frame');
      if (plan.kind === 'scale' && (plan.width <= 0 || plan.height <= 0 || !Number.isInteger(plan.width) || !Number.isInteger(plan.height)))
        return fail('failed', 'scale dimensions must be positive integers');
      const argv = [this.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', await this.path(plan.input)];
      if (plan.kind === 'trim') argv.push('-ss', String(plan.start), '-t', String(plan.end - plan.start));
      if (plan.kind === 'crop') argv.push('-vf', `crop=${plan.width}:${plan.height}:${plan.x}:${plan.y}`);
      if (plan.kind === 'scale') argv.push('-vf', `scale=${plan.width}:${plan.height}`);
      argv.push('-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p');
      argv.push(...(plan.keep_audio && source.has_audio ? ['-c:a', 'aac'] : ['-an']), output);
      const result = await run(argv, { timeoutMs: this.timeoutMs });
      if (result.timedOut || result.signal) return fail('unknown', 'render interrupted; output may exist');
      if (result.code !== 0) return fail('failed', result.output, result.code);
      const digest = await sha256(await this.path(plan.output));
      this.events.push({ operation: 'media.render', status: 'ok', input: plan.input, output: plan.output,
        input_sha256: source.sha256, output_sha256: digest, plan });
      return { status: 'ok', output: plan.output, exit_code: 0, sha256: digest, detail: '' };
    } catch (error) { return fail('failed', error instanceof Error ? error.message : String(error)); }
  }

  async inspect(request: MediaRequest, plan: Plan, receipt: Receipt): Promise<Inspection> {
    const blank = (status: Inspection['status'], detail: string): Inspection => ({ status, width: 0, height: 0, duration: 0,
      has_audio: false, sha256: '', visual_status: 'unavailable', visual_detail: '', detail });
    if (receipt.status !== 'ok') return blank('unavailable', receipt.detail);
    const clip = await this.probe(receipt.output);
    if (clip.status !== 'ok') return blank('failed', clip.detail);
    let visual_status: Inspection['visual_status'] = plan.kind === 'crop' ? 'unavailable' : 'not-needed', visual_detail = '';
    if (plan.kind === 'crop' && this.vision) {
      const folder = await mkdtemp(join(tmpdir(), 'natlang-media-frame-'));
      try {
        const frame = join(folder, 'frame.png');
        const sample = await run([this.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-ss', String(clip.duration / 2),
          '-i', await this.path(receipt.output), '-frames:v', '1', frame], { timeoutMs: this.timeoutMs });
        if (sample.code !== 0 || sample.timedOut) throw new Error(`frame sampling failed: ${sample.output}`);
        const verdict = await this.vision({ frame, request: request.text, plan, output_sha256: clip.sha256 });
        visual_status = verdict.status === 'supported' || verdict.status === 'contradicted' ? verdict.status : 'uncertain';
        visual_detail = String(verdict.detail ?? '');
        this.events.push({ operation: 'media.vision', status: visual_status, model_id: String(verdict.model_id ?? ''), output_sha256: clip.sha256 });
      } catch (error) {
        visual_status = 'unavailable'; visual_detail = error instanceof Error ? error.message : String(error);
      } finally { await rm(folder, { recursive: true, force: true }); }
    }
    return { status: 'ok', width: clip.width, height: clip.height, duration: clip.duration, has_audio: clip.has_audio,
      sha256: clip.sha256, visual_status, visual_detail, detail: '' };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Interpret a media request, run one exact transform, inspect the output, and report uncertainty honestly. */
export async function transform(media: MediaWorkspace, request: MediaRequest, files?: FolderHandle): Promise<MediaResult> {
  const source = await media.probe(request.input);
  if (source.status !== 'ok') return { status: 'failed', output: request.output, technical_ok: false, semantic_ok: false,
    needs_visual_review: false, explanation: source.detail,
    receipt: { status: 'failed', output: request.output, exit_code: -1, sha256: '', detail: source.detail },
    inspection: { status: 'unavailable', width: 0, height: 0, duration: 0, has_audio: false, sha256: '',
      visual_status: 'unavailable', visual_detail: '', detail: 'source unavailable' } };
  const plan = await choose(request, source, files);
  const receipt: Receipt = plan.input !== request.input || plan.output !== request.output ?
    { status: 'failed', output: request.output, exit_code: -1, sha256: '', detail: 'plan changed the selected input or output' } :
    await media.render(plan);
  const inspection = await media.inspect(request, plan, receipt);
  const assessment = await assess(request, source, plan, receipt, inspection, files);
  const expectedWidth = plan.kind === 'crop' || plan.kind === 'scale' ? plan.width : source.width;
  const expectedHeight = plan.kind === 'crop' || plan.kind === 'scale' ? plan.height : source.height;
  const expectedDuration = plan.kind === 'trim' ? plan.end - plan.start : source.duration;
  const technical_ok = receipt.status === 'ok' && inspection.status === 'ok' && receipt.output === request.output &&
    inspection.sha256 === receipt.sha256 && inspection.width === expectedWidth && inspection.height === expectedHeight &&
    Math.abs(inspection.duration - expectedDuration) <= 0.16 && inspection.has_audio === (plan.keep_audio && source.has_audio);
  const needs_visual_review = plan.kind === 'crop' && inspection.visual_status !== 'supported' || assessment.needs_visual_review;
  const semantic_ok = technical_ok && assessment.intent_met && inspection.visual_status !== 'contradicted';
  const status: MediaResult['status'] = receipt.status === 'unknown' ? 'unknown' : receipt.status === 'unsupported' ? 'unsupported' :
    !technical_ok ? 'failed' : !semantic_ok ? 'rejected' : needs_visual_review ? 'review' : 'verified';
  return { status, output: request.output, technical_ok, semantic_ok, needs_visual_review, receipt, inspection,
    explanation: technical_ok ? assessment.explanation : `Technical check failed: ${receipt.detail || inspection.detail}` };
}
