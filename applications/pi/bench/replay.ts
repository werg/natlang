/**
 * Replay of teacher trajectories in pi's own tools (plans/neuralese/HARNESS_BENCH.md §1.3), statically: no container
 * and no model.
 *
 * A trajectory prepared by `natlang_neuralese.harness_bench.prepare` (a pi transcript of another harness's run, with
 * its repository and base commit) is walked in order over a checkout of the base commit:
 *
 * - read, edit and write run pi-durable's own tools on the checkout, so their results are exactly what pi shows, and
 *   the checkout follows the teacher's edits;
 * - a `view` without a range becomes what the checkout says it is: a read of a file, or the listing of a directory,
 *   which runs in the checkout;
 * - bash keeps the output the teacher's environment recorded (only that environment could produce it), in pi's
 *   format: the command's output, and pi's error diagnostic when it exited non-zero, bounded as pi bounds bash.
 *
 * Each step is checked against the recording: a read must show the text the teacher saw, and an edit or write must
 * succeed or fail as the teacher's did. The first disagreement means the checkout no longer matches the teacher's
 * workspace (a command changed files, or a tool's semantics differ), so replay stops there and reports the step; the
 * steps before it are verified.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative } from 'node:path';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { ToolCall, ToolResultMessage } from '@earendil-works/pi-ai';
import { NodeExecutionEnv } from '../vendor/durable/src/env/node.ts';
import { appendToolResult, boundContent, toolDiagnostic, truncated } from '../vendor/durable/src/harness/tool.ts';
import type { ToolExecutionResult } from '../vendor/durable/src/harness/types.ts';
import { createEditTool, createReadTool, createWriteTool } from '../vendor/durable/src/tools/index.ts';
import { DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES } from '../vendor/durable/src/truncate.ts';

type Part = { type: string; text?: string; id?: string; name?: string; arguments?: Record<string, unknown> };
type Message = { role: string; content: Part[] | string; toolCallId?: string; toolName?: string; isError?: boolean; [key: string]: unknown };
export type Prepared = { id: string; repo: string; cwd: string; base_commit: string; messages: Message[]; views?: string[]; lossy?: unknown[] };
export type ReplayReport = { steps: number; replayed: number; verifiedReads: number; listings: number; recordedBash: number;
  /** Steps where pi's tool answers where the teacher's refused, with the workspace still in agreement (a view range past the end). */
  semanticsDiffer?: number;
  /** Files the environment provided (absent from the repository), taken from the teacher's whole view of them. */
  seeded?: string[];
  diverged?: { message: number; call: string; reason: string } };

const context = BACKGROUND_CONTEXT;
const tools = { read: createReadTool(), edit: createEditTool(), write: createWriteTool() };
const LISTING = /^find (\S+|'[^']*') -maxdepth 2 -not -path '\*\/\.\*'$/;

/** The base commit's files in `into`, from a bare partial clone kept under `repos` (fetched once per repository). */
export function checkout(repo: string, commit: string, repos: string, into: string): void {
  mkdirSync(repos, { recursive: true });
  const bare = join(repos, `${repo.replace('/', '__')}.git`);
  if (!existsSync(bare))
    execFileSync('git', ['clone', '--quiet', '--bare', '--filter=blob:none', `https://github.com/${repo}.git`, bare], { stdio: 'ignore' });
  const has = spawnSync('git', ['-C', bare, 'cat-file', '-e', `${commit}^{commit}`]).status === 0;
  if (!has) execFileSync('git', ['-C', bare, 'fetch', '--quiet', '--filter=blob:none', 'origin', commit], { stdio: 'ignore' });
  mkdirSync(into, { recursive: true });
  const archive = spawnSync('sh', ['-c', 'git -C "$1" archive "$2" | tar -x -C "$3"', 'sh', bare, commit, into]);
  if (archive.status !== 0) throw new Error(`checkout of ${repo}@${commit} failed: ${archive.stderr.toString().slice(0, 300)}`);
}

/** The message pi stores for a tool result: its content with rendered diagnostics, as appendToolResult builds it. */
async function resultMessage(call: ToolCall, result: ToolExecutionResult): Promise<ToolResultMessage> {
  const tx = { appendEntry: async (_type: unknown, _conversation: unknown, data: { model: ToolResultMessage[] }) => data };
  const entry = await appendToolResult(tx as never, 0 as never, call, result, 0) as unknown as { model: ToolResultMessage[] };
  const { timestamp: _timestamp, ...message } = entry.model[0] as ToolResultMessage & { timestamp?: number };
  return message as ToolResultMessage;
}

/** A tool's result, or its failure as the Harness reports one. */
async function run(name: 'read' | 'edit' | 'write', args: Record<string, unknown>, env: NodeExecutionEnv): Promise<ToolExecutionResult> {
  try {
    return await (tools[name] as { execute(args: unknown, api: unknown, context: unknown): Promise<ToolExecutionResult> })
      .execute(args, { env, callId: 'replay' }, context);
  } catch (error) {
    return { isError: true, diagnostics: [toolDiagnostic('tool_error', error instanceof Error ? error.message : String(error))] };
  }
}

const text = (content: Part[] | string): string => typeof content === 'string' ? content :
  content.filter(part => part.type === 'text').map(part => part.text ?? '').join('');

const CLIPPED = '<response clipped>';

/**
 * The file text an OpenHands `cat -n` view showed (first line number and lines), or null when it was not one. A view
 * OpenHands clipped at its output limit ends at `<response clipped>`: its last line is only a prefix (`clipped`).
 */
function viewed(recorded: string): { first: number; lines: string[]; clipped: boolean } | null {
  const header = /^Here's the result of running `cat -n` on [^\n]*:\n/.exec(recorded);
  if (!header) return null;
  const lines: string[] = [];
  let first = 0, clipped = false;
  for (const line of recorded.slice(header[0].length).split('\n')) {
    const numbered = /^\s*(\d+)\t(.*)$/.exec(line);
    if (!numbered) continue;
    if (!first) first = Number(numbered[1]);
    const at = numbered[2]!.indexOf(CLIPPED);
    if (at >= 0) { lines.push(numbered[2]!.slice(0, at)); clipped = true; break; }
    lines.push(numbered[2]!);
  }
  return { first, lines, clipped };
}

/** Whether the file's lines from `shown.first` are what the view showed (its clipped last line as a prefix). */
function agrees(file: string, shown: { first: number; lines: string[]; clipped: boolean }): boolean {
  const lines = file.split('\n').slice(shown.first - 1, shown.first - 1 + shown.lines.length);
  if (lines.length < shown.lines.length) return false;
  return shown.lines.every((line, index) => shown.clipped && index === shown.lines.length - 1 ?
    lines[index]!.startsWith(line) : lines[index]!.trimEnd() === line.trimEnd());
}

/** OpenHands' bash observation as output and exit code (its trailer removed), or the exit code null when absent. */
export function recordedBash(recorded: string): { output: string; exitCode: number | null } {
  let output = recorded, exitCode: number | null = null;
  const finished = /\n?\[Command finished with exit code (-?\d+)\]\s*$/.exec(output);
  if (finished) { exitCode = Number(finished[1]); output = output.slice(0, finished.index); }
  output = output.replace(/\n?\[Python interpreter: [^\]\n]*\]\s*$/, '').replace(/\n?\[Current working directory: [^\]\n]*\]\s*$/, '');
  const completed = /\n?\[The command completed with exit code (-?\d+)\.\]\s*$/.exec(output);
  if (completed) { exitCode ??= Number(completed[1]); output = output.slice(0, completed.index); }
  return { output, exitCode };
}

/** bash's result in pi for a recorded output: the output (bounded to its tail) and pi's diagnostic for a failure. */
function bashResult(output: string, exitCode: number | null): ToolExecutionResult {
  const failed = exitCode !== null && exitCode !== 0;
  const bounded = boundContent(output === '' ? [] : [{ type: 'text', text: output }],
    { maxBytes: DEFAULT_MAX_BYTES, maxLines: DEFAULT_MAX_LINES, retain: 'tail' });
  const diagnostics = [...(failed ? [toolDiagnostic('tool_error', `Command exited with code ${exitCode}`)] : []),
    ...(bounded.droppedBytes > 0 ? [truncated(bounded, 'tail')] : [])];
  return { content: bounded.content, ...(failed ? { isError: true } : {}), ...(diagnostics.length ? { diagnostics } : {}) };
}

/** Replay one prepared trajectory in a checkout at `root`. Returns the transcript with pi's results, and the report. */
export async function replay(prepared: Prepared, root: string): Promise<{ messages: Message[]; report: ReplayReport }> {
  const env = new NodeExecutionEnv({ cwd: root });
  const inside = (path: unknown): unknown => typeof path === 'string' && isAbsolute(path) &&
    (path === prepared.cwd || path.startsWith(`${prepared.cwd}/`)) ? relative(prepared.cwd, path) || '.' : path;
  const views = new Set(prepared.views ?? []);
  const calls = new Map<string, Part>();
  const messages: Message[] = structuredClone(prepared.messages);
  const report: ReplayReport = { steps: 0, replayed: 0, verifiedReads: 0, listings: 0, recordedBash: 0 };
  const listing = (target: string): ToolExecutionResult => {
    const found = spawnSync('find', [target, '-maxdepth', '2', '-not', '-path', '*/.*'], { cwd: root, encoding: 'utf8' });
    return bashResult(found.stdout.replaceAll(root, prepared.cwd).replace(/\n$/, ''), found.status ?? 1);
  };
  for (let index = 0; index < messages.length && !report.diverged; index++) {
    const message = messages[index]!;
    if (message.role === 'assistant' && Array.isArray(message.content))
      for (const part of message.content) if (part.type === 'toolCall') calls.set(part.id!, part);
    if (message.role !== 'toolResult') continue;
    const call = calls.get(message.toolCallId!);
    if (!call) continue;
    report.steps++;
    const recorded = text(message.content);
    const diverge = (reason: string) => { report.diverged = { message: index, call: call.id!, reason }; };
    let args = { ...(call.arguments ?? {}) };
    if ('path' in args) args = { ...args, path: inside(args.path) };
    // A view without a range: the checkout says whether it read a file or listed a directory.
    if (views.has(call.id!)) {
      const target = String(call.name === 'read' ? args.path : (LISTING.exec(String(args.command))?.[1] ?? '.').replace(/^'|'$/g, ''));
      const directory = existsSync(join(root, target)) && statSync(join(root, target)).isDirectory();
      if (directory && call.name === 'read') { call.name = 'bash'; call.arguments = { command: `find ${target === '.' ? '.' : `'${target}'`} -maxdepth 2 -not -path '*/.*'` }; }
      if (!directory && call.name === 'bash') { call.name = 'read'; call.arguments = { path: target }; }
      args = { ...call.arguments };
    }
    // A file the teacher read whole that the checkout lacks, and nothing wrote before, came with its environment (OpenHands
    // writes the issue to issue.md): its content is what the view showed.
    if (call.name === 'read' && args.offset === undefined && !existsSync(join(root, String(args.path)))) {
      const shown = viewed(recorded);
      if (shown && shown.first === 1 && !shown.clipped) {
        mkdirSync(dirname(join(root, String(args.path))), { recursive: true });
        writeFileSync(join(root, String(args.path)), shown.lines.join('\n') + '\n');
        (report.seeded ??= []).push(String(args.path));
      }
    }
    let result: ToolExecutionResult;
    if (call.name === 'read' || call.name === 'edit' || call.name === 'write') {
      result = await run(call.name, args, env);
      const teacherFailed = call.name === 'read' ? !viewed(recorded) : call.name === 'edit' ? !/has been edited/.test(recorded) :
        !/File created successfully/.test(recorded);
      // OpenHands refuses a view range that ends past the file; pi's read returns the lines there are. The workspace
      // still agrees, so pi's result stands and replay goes on.
      const pastEnd = call.name === 'read' && teacherFailed && !result.isError && /Invalid `view_range` parameter: .*should be smaller than the number of lines in the file/s.test(recorded);
      if (pastEnd) report.semanticsDiffer = (report.semanticsDiffer ?? 0) + 1;
      else if (teacherFailed !== Boolean(result.isError)) { diverge(`${call.name} ${teacherFailed ? 'failed for the teacher' : 'failed in pi'}`); break; }
      if (call.name === 'read' && !teacherFailed) {
        if (!agrees(readFileSync(join(root, String(args.path)), 'utf8'), viewed(recorded)!)) {
          diverge('read shows other text than the teacher saw'); break;
        }
        report.verifiedReads++;
      }
    } else if (call.name === 'bash' && views.has(call.id!)) {
      result = listing(String(LISTING.exec(String(call.arguments?.command))?.[1] ?? '.').replace(/^'|'$/g, ''));
      report.listings++;
    } else if (call.name === 'bash') {
      const { output, exitCode } = recordedBash(recorded);
      result = bashResult(output, exitCode);
      report.recordedBash++;
    } else { diverge(`${call.name} has no pi tool`); break; }
    const replaced = await resultMessage({ type: 'toolCall', id: call.id!, name: call.name!, arguments: call.arguments ?? {} } as ToolCall, result);
    messages[index] = { ...replaced, toolName: call.name } as unknown as Message;
    report.replayed++;
  }
  return { messages, report };
}

/** Replay every prepared trajectory of `input` (JSON lines) into `output`, each in a fresh checkout. */
export async function replayFile(input: string, output: string, repos: string, log: (line: string) => void): Promise<void> {
  const { appendFileSync } = await import('node:fs');
  writeFileSync(output, '');
  const totals = { trajectories: 0, fullyVerified: 0, diverged: 0, failed: 0 };
  for (const line of readFileSync(input, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    const prepared = JSON.parse(line) as Prepared;
    totals.trajectories++;
    const root = mkdtempSync(join(tmpdir(), 'pi-replay-'));
    try {
      checkout(prepared.repo, prepared.base_commit, repos, root);
      const { messages, report } = await replay(prepared, root);
      if (report.diverged) totals.diverged++; else totals.fullyVerified++;
      appendFileSync(output, JSON.stringify({ ...prepared, messages, replay: report }) + '\n');
      log(`${prepared.id} ${prepared.repo}: ${report.replayed}/${report.steps} steps${report.diverged ? `, diverged at message ${report.diverged.message}: ${report.diverged.reason}` : ''}`);
    } catch (error) {
      totals.failed++;
      log(`${prepared.id} ${prepared.repo}: ${error instanceof Error ? error.message : String(error)}`);
    } finally { rmSync(root, { recursive: true, force: true }); }
  }
  log(JSON.stringify(totals));
}
