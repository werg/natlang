/**
 * `natlang run applications/pi -- [options] TASK...`: pi's coding agent on the workspace, in print mode. The harness is
 * pi-durable with natural-language task kinds (index.ts); the launcher's model is the executor that runs the
 * natural-language functions; the agent is a provider model reached through pi-ai.
 *
 * Options:
 *   --agent-endpoint URL --agent-model ID   an OpenAI-compatible server for the agent (default: the executor's
 *                                           profile endpoint and model); --agent-key-env VAR for its key;
 *                                           --context-window N (default 65536), --max-tokens N (default 16384)
 *   --thinking LEVEL                        the agent's thinking level (default off)
 *   --cwd DIR                               the agent's working directory (default: the workspace)
 *   --session FILE                          the SQLite session (default: a new one under the state directory)
 *   --context natural-language              context building in natural language (default crisp)
 *   --scheduler natural-language            the scheduler's policy in natural language (default crisp)
 *   --admission natural-language            admission in natural language (default crisp)
 *   --pure                                  all three in natural language
 *   --executor-context N                    the executor's context budget in tokens (default 57344; natlang's
 *                                           default of 16384 makes the harness functions compact constantly)
 *   --quiet                                 no phase log on stderr
 * `natlang run applications/pi -- eval [NAME...] [--out DIR] [--minutes N] [options]`: the tasks in tasks/, each on a
 * fresh git copy of its repository, judged by its check command.
 */
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import { createModels, createProvider, type AssistantMessage, type Models } from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { NatlangRuntime, type TargetContext } from '@natlang/node';
import { openNodeSqliteStorage } from './vendor/durable/src/storage/sqlite/node.ts';
import type { EntryId } from './vendor/durable/src/types.ts';
import { codingRegistry, createEnvs } from './extensions/index.ts';
import { openPi, type Implementation } from './index.ts';

const context = BACKGROUND_CONTEXT;
const VALUED = ['--executor-context', '--agent-endpoint', '--agent-model', '--agent-key-env', '--context-window', '--max-tokens', '--thinking', '--cwd',
  '--session', '--context', '--scheduler', '--admission', '--out', '--minutes'];
const option = (args: string[], name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
const positional = (args: string[]) => args.filter((arg, i) => !arg.startsWith('-') && !VALUED.includes(args[i - 1] ?? ''));
const taskDirectory = ['../../tasks', '../../pi/tasks', './tasks'].map(path => fileURLToPath(new URL(path, import.meta.url))).find(path => existsSync(path))!;

/** The launcher's profile endpoint, for an agent on the same server as the executor. */
function profileEndpoint(): { endpoint: string; model: string } | undefined {
  try {
    const home = process.env.HOME ?? '';
    const config = JSON.parse(readFileSync(join(home, '.config', 'natlang', 'config.json'), 'utf8')) as
      { defaultProfile?: string; profiles?: Record<string, { endpoint: string; model: string }> };
    const profile = config.profiles?.[process.env.NATLANG_PROFILE ?? config.defaultProfile ?? ''];
    return profile ? { endpoint: profile.endpoint, model: profile.model } : undefined;
  } catch { return undefined; }
}

/** pi-ai models with one OpenAI-compatible provider, "agent", serving the agent model. */
export function agentModels(args: string[]): { models: Models; ref: { provider: string; modelId: string } } {
  const profile = profileEndpoint();
  const endpoint = option(args, '--agent-endpoint') ?? process.env.PI_AGENT_ENDPOINT ?? profile?.endpoint;
  const modelId = option(args, '--agent-model') ?? process.env.PI_AGENT_MODEL ?? profile?.model;
  if (!endpoint || !modelId) throw new Error('name the agent model: --agent-endpoint URL --agent-model ID');
  const keyVariable = option(args, '--agent-key-env');
  const baseUrl = `${endpoint.replace(/\/+$/, '').replace(/\/v1$/, '')}/v1`;
  const models = createModels();
  models.setProvider(createProvider({
    id: 'agent', name: 'Agent', baseUrl,
    auth: { apiKey: { name: 'agent key', resolve: async () => ({ auth: { apiKey: (keyVariable ? process.env[keyVariable] : undefined) ?? 'none' } }) } },
    models: [{ id: modelId, name: modelId, api: 'openai-completions', provider: 'agent', baseUrl, input: ['text'], reasoning: true,
      contextWindow: Number(option(args, '--context-window') ?? 65536), maxTokens: Number(option(args, '--max-tokens') ?? 16384),
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { thinkingFormat: 'qwen-chat-template', supportsDeveloperRole: false, supportsStore: false, supportsReasoningEffort: false,
        maxTokensField: 'max_tokens' } } as never],
    api: openAICompletionsApi(),
  }));
  return { models, ref: { provider: 'agent', modelId } };
}

function implementations(args: string[]): { context: Implementation; scheduler: Implementation; admission: Implementation } {
  const pick = (name: string): Implementation => args.includes('--pure') || option(args, `--${name}`) === 'natural-language' ? 'natural-language' : 'crisp';
  return { context: pick('context'), scheduler: pick('scheduler'), admission: pick('admission') };
}

const textOf = (message: AssistantMessage | undefined) =>
  (message?.content ?? []).flatMap(item => item.type === 'text' ? [item.text] : []).join('\n').trim();

/** The launcher's runtime with the executor's context budget set. */
function executor(target: TargetContext, args: string[]): NatlangRuntime {
  const configured = target.runtime.options.model;
  const base = typeof configured === 'object' && configured ? configured : {};
  const driver = typeof configured === 'function' ? configured : (configured as { driver?: unknown } | undefined)?.driver ?? target.model;
  return new NatlangRuntime({ ...target.runtime.options,
    model: { ...base, driver, contextTokens: Number(option(args, '--executor-context') ?? 57344) } as never });
}

export type RunResult = { status: string; answer: string; reason?: string; ms: number };

/** Run one task to its answer on a session at `sessionPath`. */
export async function runTask(target: TargetContext, args: string[], task: string, cwd: string, sessionPath: string, signal?: AbortSignal): Promise<RunResult> {
  const started = Date.now();
  const { models, ref } = agentModels(args);
  const quiet = args.includes('--quiet');
  const log = (line: string) => { if (!quiet) target.io.error.write(`${line}\n`); };
  const envs = createEnvs(cwd);
  const natlang = executor(target, args);
  mkdirSync(join(sessionPath, '..'), { recursive: true });
  const harness = await openPi({
    storage: await openNodeSqliteStorage(sessionPath),
    natlang,
    models,
    registry: codingRegistry(natlang, { cwd }),
    env: envs.env,
    implementations: implementations(args),
    onReport: error => log(`  [report] ${error instanceof Error ? error.message : String(error)}`),
    onPhase: event => log(`  ${event.kind}#${event.taskId} ${event.phase}${event.mode === 'abort' ? ' (abort)' : ''}` +
      `${event.attempt > 1 ? ` attempt ${event.attempt}` : ''}: ${event.error ? `failed: ${event.error.slice(0, 300)}` : (event.summary ?? '').slice(0, 200)}`),
  }, context);
  const onAbort = () => { void harness.root(context).then(root => root.abort(context)).catch(() => {}); };
  signal?.addEventListener('abort', onAbort, { once: true });
  try {
    const root = await harness.root(context, { agent: { model: ref, thinkingLevel: (option(args, '--thinking') ?? 'off') as never, cwd } });
    const submission = await root.submit({ type: 'input', content: task }, context);
    const settled = await submission.wait(context);
    if (settled.status !== 'done') return { status: settled.status, answer: '', reason: settled.reason, ms: Date.now() - started };
    const entry = (await root.entries({ minEntryId: settled.answer as EntryId, maxEntryId: settled.answer as EntryId }, 1, undefined, context)).items[0];
    return { status: 'done', answer: textOf(entry?.model?.[0] as AssistantMessage | undefined), ms: Date.now() - started };
  } finally {
    signal?.removeEventListener('abort', onAbort);
    await harness.close(context);
    await envs.cleanup(context);
  }
}

export async function main(target: TargetContext): Promise<number> {
  const args = target.args;
  if (args[0] === 'eval') return evaluate(target, args.slice(1));
  const task = positional(args).join(' ').trim();
  if (!task) { target.io.error.write('usage: natlang run applications/pi -- [options] TASK...\n'); return 2; }
  const cwd = resolve(target.workspace, option(args, '--cwd') ?? '.');
  const session = option(args, '--session') ? resolve(target.workspace, option(args, '--session')!) :
    join(target.stateDirectory, 'sessions', `${new Date().toISOString().replace(/[:.]/g, '-')}.sqlite`);
  const result = await runTask(target, args, task, cwd, session);
  target.io.output.write(`${result.answer || `(${result.status}${result.reason ? `: ${result.reason}` : ''})`}\n`);
  return result.status === 'done' ? 0 : 1;
}

type TaskSpec = { prompt: string; check: string };

/** Each task on a fresh git copy of its repository, judged by its check command. */
async function evaluate(target: TargetContext, args: string[]): Promise<number> {
  const names = positional(args);
  const out = resolve(target.workspace, option(args, '--out') ?? 'pi-eval-out');
  mkdirSync(out, { recursive: true });
  const tasks = names.length ? names.filter(name => readdirSync(taskDirectory).includes(name)) : readdirSync(taskDirectory).sort();
  const minutes = Number(option(args, '--minutes') ?? 0);
  const rows: Record<string, unknown>[] = [];
  for (const name of tasks) {
    const spec = JSON.parse(readFileSync(join(taskDirectory, name, 'task.json'), 'utf8')) as TaskSpec;
    const cwd = join(mkdtempSync(join(tmpdir(), `pi-${name}-`)), 'repo');
    cpSync(join(taskDirectory, name, 'repo'), cwd, { recursive: true });
    execFileSync('sh', ['-c', 'git init -q && git add -A && git -c user.name=pi -c user.email=pi@example.com commit -qm start'], { cwd });
    target.io.error.write(`\n=== ${name}\n`);
    let result: RunResult | undefined, error: string | undefined;
    try {
      const limit = minutes > 0 ? AbortSignal.timeout(minutes * 60_000) : undefined;
      result = await runTask(target, args, spec.prompt, cwd, join(out, `${name}.sqlite`), limit);
    } catch (caught) { error = caught instanceof Error ? caught.message : String(caught); }
    let passed = false, check = '';
    try { check = execFileSync('sh', ['-c', spec.check.replaceAll('{task}', join(taskDirectory, name))], { cwd, encoding: 'utf8', timeout: 300_000 }); passed = true; }
    catch (failure) { check = String((failure as { stdout?: string }).stdout ?? '') + String((failure as { stderr?: string }).stderr ?? ''); }
    const row = { task: name, passed, status: result?.status, reason: result?.reason, ms: result?.ms, error, answer: result?.answer.slice(0, 2000), check: check.slice(-1500) };
    rows.push(row);
    target.io.error.write(`=== ${name}: ${passed ? 'PASS' : 'FAIL'}${error ? `: ${error.slice(0, 300)}` : ''}\n`);
    writeFileSync(join(out, 'results.json'), JSON.stringify(rows, null, 2));
  }
  target.io.output.write(`${rows.filter(row => row.passed).length}/${rows.length} passed\n`);
  return 0;
}
