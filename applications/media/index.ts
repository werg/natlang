/**
 * Semantic media workbench. `transform` probes the source, lets natlang choose one trim/crop/scale/
 * transcode operation and its parameters (the host assembles the plan and checks it, returning a problem to the
 * parameter stage once), renders it with FFmpeg, inspects the output, has natlang assess intent when the exact
 * checks passed, and finalizes only when exact metadata checks agree. `MediaWorkspace` keeps media bytes, sampled frames
 * and FFmpeg processes on the host; an optional `vision` callback judges a sampled frame for crops.
 */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream } from 'node:fs';
import { lstat, mkdtemp, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { extname, isAbsolute, join, relative, resolve } from 'node:path';
import { builtin, untrusted, type FolderHandle, type Untrusted } from '@natlang/node';
import chooseOperation from './chooseOperation.nl';
import planTrim from './planTrim.nl';
import planCrop from './planCrop.nl';
import planScale from './planScale.nl';
import planTranscode from './planTranscode.nl';
import keepAudio from './keepAudio.nl';
import needsPicture from './needsPicture.nl';
import assessIntent from './assessIntent.nl';
import type { Clip, Container, Inspection, MediaRequest, Operation, Plan, PlanFacts, Receipt, SourceFacts, VideoCodec } from './types.js';

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

/** What the render service writes: a container's FFmpeg muxer, file extensions, audio codec and the video codecs it can hold. */
export const MEDIA_CONTAINERS: Record<Container, { muxer: string, extensions: string[], audio: string, codecs: VideoCodec[] }> = {
  mp4: { muxer: 'mp4', extensions: ['.mp4', '.m4v'], audio: 'aac', codecs: ['h264', 'h265', 'vp9', 'av1'] },
  mkv: { muxer: 'matroska', extensions: ['.mkv'], audio: 'aac', codecs: ['h264', 'h265', 'vp9', 'av1'] },
  webm: { muxer: 'webm', extensions: ['.webm'], audio: 'libopus', codecs: ['vp9', 'av1'] },
  mov: { muxer: 'mov', extensions: ['.mov'], audio: 'aac', codecs: ['h264', 'h265'] },
};
/** The FFmpeg encoder, its fixed arguments and the codec name ffprobe reports for each video codec. Fixed settings, not model output. */
export const MEDIA_ENCODERS: Record<VideoCodec, { encoder: string, args: string[], probe: string }> = {
  h264: { encoder: 'libx264', args: ['-preset', 'ultrafast'], probe: 'h264' },
  h265: { encoder: 'libx265', args: ['-preset', 'ultrafast', '-x265-params', 'log-level=error'], probe: 'hevc' },
  vp9: { encoder: 'libvpx-vp9', args: ['-deadline', 'realtime', '-cpu-used', '8', '-b:v', '0', '-crf', '32'], probe: 'vp9' },
  av1: { encoder: 'libsvtav1', args: ['-preset', '12'], probe: 'av1' },
};
/** Duration tolerance of the technical check, seconds: encoder frame rounding measured on short clips. */
export const DURATION_TOLERANCE_SECONDS = 0.16;
const OPERATIONS = ['trim', 'crop', 'scale', 'transcode'];
const DEFAULT_CODEC: Record<Container, VideoCodec> = { mp4: 'h264', mkv: 'h264', webm: 'vp9', mov: 'h264' };

/** The container whose file extension an output name carries, or undefined for an extension no container uses. */
export function containerOf(name: string): Container | undefined {
  const extension = extname(name).toLowerCase();
  return (Object.keys(MEDIA_CONTAINERS) as Container[]).find(container => MEDIA_CONTAINERS[container].extensions.includes(extension));
}

/** A reason a plan cannot be rendered. `parameters` problems return to the parameter stage; `output` problems belong to the request. */
export type PlanProblem = { text: string, stage: 'parameters' | 'output' };

/**
 * The exact validity rules of a plan against its source clip; null when the plan can be rendered. A plan of kind
 * `unsupported` has nothing to check. The one place these rules live: `transform` calls it before the render so
 * its problem text can return to the parameter stage, and `render` calls it again.
 */
export function checkPlan(plan: Plan, source: Pick<Clip, 'width' | 'height' | 'duration'>): PlanProblem | null {
  if (plan.kind === 'unsupported') return null;
  const problem = (text: string): PlanProblem => ({ text, stage: 'parameters' });
  if (typeof plan.keep_audio !== 'boolean') return problem('keep_audio must be boolean');
  if (!([plan.start, plan.end, plan.x, plan.y, plan.width, plan.height]).every(Number.isFinite))
    return problem('plan dimensions and times must be finite');
  const container = MEDIA_CONTAINERS[plan.container];
  if (!container) return problem(`container must be one of ${Object.keys(MEDIA_CONTAINERS).join(', ')}`);
  if (!MEDIA_ENCODERS[plan.video_codec]) return problem(`video codec must be one of ${Object.keys(MEDIA_ENCODERS).join(', ')}`);
  if (!container.codecs.includes(plan.video_codec))
    return problem(`${plan.container} holds the video codecs ${container.codecs.join(', ')}, and the plan asks for ${plan.video_codec}`);
  if (containerOf(plan.output) !== plan.container)
    return { stage: plan.kind === 'transcode' ? 'parameters' : 'output', text: `the output name ${plan.output} must end in ${container.extensions.join(' or ')} for container ${plan.container}` };
  if (plan.kind === 'trim' && (plan.start < 0 || plan.end <= plan.start || plan.end > source.duration + 0.02))
    return problem(`trim interval is outside source duration (0 to ${source.duration} seconds)`);
  if (plan.kind === 'crop' && (plan.x < 0 || plan.y < 0 || plan.width <= 0 || plan.height <= 0 ||
      plan.x + plan.width > source.width || plan.y + plan.height > source.height ||
      [plan.x, plan.y, plan.width, plan.height].some(n => !Number.isInteger(n))))
    return problem(`crop rectangle is outside source frame (${source.width} by ${source.height} pixels, whole numbers)`);
  if (plan.kind === 'scale' && (plan.width <= 0 || plan.height <= 0 || !Number.isInteger(plan.width) || !Number.isInteger(plan.height)))
    return problem('scale dimensions must be positive integers');
  if ((plan.kind === 'crop' || plan.kind === 'scale') && (plan.width % 2 !== 0 || plan.height % 2 !== 0))
    return problem('the output width and height must be even numbers of pixels for the yuv420p video format');
  return null;
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
  /** Seconds of duration difference the technical check accepts. */
  readonly durationTolerance: number;
  private readonly maxOutput: number;
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, { ffmpeg = 'ffmpeg', ffprobe = 'ffprobe', vision = null as VisionInspector | null, timeoutMs = 120_000,
      durationTolerance = DURATION_TOLERANCE_SECONDS, maxOutput = 8192 } = {}) {
    this.rootPath = resolve(root);
    this.ffmpeg = ffmpeg; this.ffprobe = ffprobe; this.vision = vision; this.timeoutMs = timeoutMs;
    this.durationTolerance = durationTolerance; this.maxOutput = maxOutput;
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
        has_audio: parsed.streams!.some(stream => stream.codec_type === 'audio'), video_codec: String(video.codec_name ?? ''), sha256: await sha256(path), detail: '' };
      this.events.push({ operation: 'media.probe', id, status: 'ok', sha256: clip.sha256 });
      return clip;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      this.events.push({ operation: 'media.probe', id, status: 'failed', detail });
      return { id, status: 'failed', width: 0, height: 0, duration: 0, has_audio: false, video_codec: '', sha256: '', detail };
    }
  }

  /** Render a plan. A plan without `container` or `video_codec` gets the output name's container and its usual codec. */
  async render(given: Omit<Plan, 'container' | 'video_codec'> & Partial<Pick<Plan, 'container' | 'video_codec'>>): Promise<Receipt> {
    const container = given.container ?? containerOf(given.output) ?? 'mp4';
    const plan: Plan = { ...given, container, video_codec: given.video_codec ?? DEFAULT_CODEC[container] };
    const fail = (status: Receipt['status'], detail: string, exit_code = -1): Receipt => {
      this.events.push({ operation: 'media.render', status, input: plan.input, output: plan.output, detail });
      return { status, output: plan.output, exit_code, sha256: '', detail };
    };
    try {
      const source = await this.probe(plan.input);
      if (source.status !== 'ok') return fail('failed', source.detail);
      const output = await this.path(plan.output, { output: true });
      if (!OPERATIONS.includes(plan.kind)) return fail('unsupported', `unsupported transform: ${plan.kind}`);
      const problem = checkPlan(plan, source);
      if (problem) return fail('failed', problem.text);
      const argv = [this.ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', await this.path(plan.input)];
      if (plan.kind === 'trim') argv.push('-ss', String(plan.start), '-t', String(plan.end - plan.start));
      if (plan.kind === 'crop') argv.push('-vf', `crop=${plan.width}:${plan.height}:${plan.x}:${plan.y}`);
      if (plan.kind === 'scale') argv.push('-vf', `scale=${plan.width}:${plan.height}`);
      const encoder = MEDIA_ENCODERS[plan.video_codec], muxer = MEDIA_CONTAINERS[plan.container];
      argv.push('-c:v', encoder.encoder, ...encoder.args, '-pix_fmt', 'yuv420p');
      argv.push(...(plan.keep_audio && source.has_audio ? ['-c:a', muxer.audio] : ['-an']), '-f', muxer.muxer, output);
      const result = await run(argv, { timeoutMs: this.timeoutMs, maxOutput: this.maxOutput });
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
      has_audio: false, video_codec: '', sha256: '', visual_status: 'unavailable', visual_detail: '', detail });
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
      video_codec: clip.video_codec, sha256: clip.sha256, visual_status, visual_detail, detail: '' };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

const factsOf = (source: Clip): SourceFacts => ({ width: source.width, height: source.height, duration: source.duration, has_audio: source.has_audio });
type Stage = (...args: unknown[]) => Promise<Partial<Plan>>;

/** The plan the host writes from the stages' answers: identifiers from the request, zeros for unused fields. */
export function assemblePlan(request: MediaRequest, kind: Operation, keep_audio: boolean, parameters: Partial<Plan> = {}): Plan {
  const container = containerOf(request.output) ?? 'mp4';
  return { kind, input: request.input, output: request.output, start: 0, end: 0, x: 0, y: 0, width: 0, height: 0,
    keep_audio, container, video_codec: DEFAULT_CODEC[container], ...parameters };
}

/** The sidecar note a request names, or an empty note when there is no folder to read it from. */
const noteFor = async (request: MediaRequest, files?: FolderHandle): Promise<Untrusted<string>> =>
  files ? await builtin('readNote')(request.text, files) as Untrusted<string> : untrusted('', 'note');

/**
 * Plan one request in stages: the operation, its parameters and the audio decision come from natlang; the host writes the
 * plan and checks it. A parameter problem returns to the parameter stage once, with the problem text.
 */
export async function planRequest(request: MediaRequest, source: Clip, note: Untrusted<string>): Promise<Plan> {
  const facts = factsOf(source);
  const operation = await chooseOperation(request.text, facts, note);
  if (!OPERATIONS.includes(operation)) return assemblePlan(request, 'unsupported', source.has_audio);
  const keep_audio = source.has_audio && await keepAudio(request.text, facts);
  const extension = extname(request.output).slice(1).toLowerCase();
  const parameters = (problem?: string): Promise<Partial<Plan>> => {
    const extra = problem === undefined ? [] : [problem];
    if (operation === 'trim') return (planTrim as Stage)(request.text, facts, note, ...extra);
    if (operation === 'crop') return (planCrop as Stage)(request.text, facts, note, ...extra);
    if (operation === 'scale') return (planScale as Stage)(request.text, facts, note, ...extra);
    return (planTranscode as Stage)(request.text, extension, facts, note, ...extra);
  };
  let plan = assemblePlan(request, operation, keep_audio, await parameters());
  const found = checkPlan(plan, source);
  if (found?.stage === 'parameters') plan = assemblePlan(request, operation, keep_audio, await parameters(found.text));
  return plan;
}

/** Interpret a media request, run one exact transform, inspect the output, and report uncertainty honestly. */
export async function transform(media: MediaWorkspace, request: MediaRequest, files?: FolderHandle): Promise<MediaResult> {
  const source = await media.probe(request.input);
  if (source.status !== 'ok') return { status: 'failed', output: request.output, technical_ok: false, semantic_ok: false,
    needs_visual_review: false, explanation: source.detail,
    receipt: { status: 'failed', output: request.output, exit_code: -1, sha256: '', detail: source.detail },
    inspection: { status: 'unavailable', width: 0, height: 0, duration: 0, has_audio: false, video_codec: '', sha256: '',
      visual_status: 'unavailable', visual_detail: '', detail: 'source unavailable' } };
  const note = await noteFor(request, files);
  const plan = await planRequest(request, source, note);
  const receipt = await media.render(plan);
  const inspection = await media.inspect(request, plan, receipt);
  const expectedWidth = plan.kind === 'crop' || plan.kind === 'scale' ? plan.width : source.width;
  const expectedHeight = plan.kind === 'crop' || plan.kind === 'scale' ? plan.height : source.height;
  const expectedDuration = plan.kind === 'trim' ? plan.end - plan.start : source.duration;
  const technical_ok = receipt.status === 'ok' && inspection.status === 'ok' && receipt.output === request.output &&
    inspection.sha256 === receipt.sha256 && inspection.width === expectedWidth && inspection.height === expectedHeight &&
    Math.abs(inspection.duration - expectedDuration) <= media.durationTolerance &&
    inspection.has_audio === (plan.keep_audio && source.has_audio) && inspection.video_codec === MEDIA_ENCODERS[plan.video_codec]?.probe;
  const cropUnseen = plan.kind === 'crop' && inspection.visual_status !== 'supported';
  let intent_met = false, needs_visual_review = cropUnseen;
  let explanation = `Technical check failed: ${receipt.detail || inspection.detail || 'the inspected clip differs from the plan'}`;
  if (technical_ok) {
    // The semantic stages run only on a clip that passed the exact checks; they read measurements, not file names or hashes.
    const observed = { width: inspection.width, height: inspection.height, duration: inspection.duration, has_audio: inspection.has_audio,
      video_codec: inspection.video_codec, visual_status: inspection.visual_status, visual_detail: inspection.visual_detail };
    const { kind, start, end, x, y, width, height, keep_audio, container, video_codec } = plan;
    const planFacts: PlanFacts = { kind, start, end, x, y, width, height, keep_audio, container, video_codec };
    const [pictureMatters, assessment] = await Promise.all([needsPicture(request.text),
      assessIntent(request.text, factsOf(source), planFacts, observed, note)]);
    intent_met = assessment.intent_met; explanation = assessment.explanation;
    needs_visual_review = cropUnseen || pictureMatters && (inspection.visual_status === 'unavailable' || inspection.visual_status === 'uncertain');
  }
  const semantic_ok = technical_ok && intent_met && inspection.visual_status !== 'contradicted';
  const status: MediaResult['status'] = receipt.status === 'unknown' ? 'unknown' : receipt.status === 'unsupported' ? 'unsupported' :
    !technical_ok ? 'failed' : !semantic_ok ? 'rejected' : needs_visual_review ? 'review' : 'verified';
  return { status, output: request.output, technical_ok, semantic_ok, needs_visual_review, receipt, inspection, explanation };
}
