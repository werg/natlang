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
 *   --agent-transport natlang               reach the agent model through natlang's own model transport, which speaks
 *                                           the Neuralese wire standard as natlang programs do (host/natlang-provider.ts;
 *                                           default: pi-ai's OpenAI-compatible provider)
 *   --agent-reader text|DIALECT             with --agent-transport natlang: what the agent model reads (default text);
 *                                           a Neuralese dialect is checked at startup against the server's
 *                                           /v1/neuralese/info, and Neuralese parts then reach it as blocks
 *   --cwd DIR                               the agent's working directory (default: the workspace)
 *   --session FILE                          the SQLite session (default: a new one under the state directory)
 *   Pluggable hot paths, each crisp|nl|shadow (default crisp; nl runs the natural-language functions, shadow runs
 *   both, uses the natural-language result and records whether they agree; natural-language is accepted for nl):
 *   --context MODE                          context building
 *   --scheduler MODE                        the scheduler's policy
 *   --admission crisp|nl                    admission (no shadow: both sides commit)
 *   --planning MODE                         the system-entry plan and the context estimate
 *   --shaping MODE                          with --companion: how long tool outputs are shortened (crisp: head and
 *                                           tail; nl: by judgment)
 *   --pure                                  nl for every point not set explicitly (with --companion, shaping too)
 *   --companion                             run the companion beside the agent (COMPANION.md): background briefings
 *   --executor-context N                    the executor's context budget in tokens (default: natlang's, sized from
 *                                           the window the executor's server reports)
 *   --quiet                                 no phase log on stderr
 * `natlang run applications/pi -- eval [NAME...] [--out DIR] [--minutes N] [options]`: the tasks in tasks/, each on a
 * fresh git copy of its repository, judged by its check command.
 * `natlang run applications/pi -- replay --in PREPARED.jsonl --out REPLAYED.jsonl [--repos DIR]`: teacher trajectories
 * replayed in pi's own tools over their base commits (bench/replay.ts), for training records.
 * `natlang run applications/pi -- surface [--cwd DIR] [--companion]`: the agent's system prompt and tool schemas as JSON
 * (surface.ts), for training records built from other agents' trajectories.
 */
import { cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import { createModels, createProvider, type AssistantMessage, type Models } from '@earendil-works/pi-ai';
import { openAICompletionsApi } from '@earendil-works/pi-ai/api/openai-completions.lazy';
import { NatlangRuntime, neuraleseServerModelTurn, openAICompatibleModelTurn, pluggableMode, type NeuraleseStore, type PluggableMode, type TargetContext } from '@natlang/node';
import { declareReader, natlangApi, type AgentReader, type ModelDriver } from './host/natlang-provider.ts';
import { openNodeSqliteStorage } from './vendor/durable/src/storage/sqlite/node.ts';
import type { EntryId } from './vendor/durable/src/types.ts';
import { codingRegistry, createEnvs } from './extensions/index.ts';
import { companion } from './extensions/companion/index.ts';
import { agentSurface } from './surface.ts';
import { replayFile } from './bench/replay.ts';
import type { Harness } from './vendor/durable/src/harness/harness.ts';
import { openPi, type Implementations } from './index.ts';

const context = BACKGROUND_CONTEXT;
const VALUED = ['--executor-context', '--agent-endpoint', '--agent-model', '--agent-key-env', '--context-window', '--max-tokens', '--thinking', '--cwd',
  '--agent-transport', '--agent-reader',
  '--session', '--context', '--scheduler', '--admission', '--planning', '--shaping', '--out', '--minutes', '--in', '--repos'];
const option = (args: string[], name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
const positional = (args: string[]) => args.filter((arg, i) => !arg.startsWith('-') && !VALUED.includes(args[i - 1] ?? ''));
const taskDirectory = ['../../tasks', '../../pi/tasks', './tasks'].map(path => fileURLToPath(new URL(path, import.meta.url))).find(path => existsSync(path))!;

/** The default profile's endpoint, for an agent on the same server as the executor when the launcher names none. */
function profileEndpoint(): { endpoint: string; model: string } | undefined {
  try {
    const home = process.env.HOME ?? '';
    const config = JSON.parse(readFileSync(join(home, '.config', 'natlang', 'config.json'), 'utf8')) as
      { defaultProfile?: string; profiles?: Record<string, { endpoint: string; model: string }> };
    const profile = config.profiles?.[process.env.NATLANG_PROFILE ?? config.defaultProfile ?? ''];
    return profile ? { endpoint: profile.endpoint, model: profile.model } : undefined;
  } catch { return undefined; }
}

/**
 * pi-ai models with one provider, "agent", serving the agent model: pi-ai's OpenAI-compatible provider, or with
 * `--agent-transport natlang` natlang's own transport, whose model declares its reader (`--agent-reader`, checked
 * against the server here). `store`: the runtime's Neuralese store, which a Neuralese server's blocks are archived in.
 */
export async function agentModels(args: string[], launcher?: { endpoint: string; model: string }, store?: NeuraleseStore):
    Promise<{ models: Models; ref: { provider: string; modelId: string } }> {
  // The launcher's own endpoint (the profile `natlang run --profile` selected) before the configured default profile.
  const profile = launcher ?? profileEndpoint();
  const endpoint = option(args, '--agent-endpoint') ?? process.env.PI_AGENT_ENDPOINT ?? profile?.endpoint;
  const modelId = option(args, '--agent-model') ?? process.env.PI_AGENT_MODEL ?? profile?.model;
  if (!endpoint || !modelId) throw new Error('name the agent model: --agent-endpoint URL --agent-model ID');
  const keyVariable = option(args, '--agent-key-env');
  const apiKey = keyVariable ? process.env[keyVariable] : undefined;
  const root = endpoint.replace(/\/+$/, '').replace(/\/v1$/, '');
  const baseUrl = `${root}/v1`;
  const transport = option(args, '--agent-transport') ?? 'pi-ai';
  if (transport !== 'pi-ai' && transport !== 'natlang') throw new Error(`--agent-transport is pi-ai or natlang, not ${transport}`);
  if (transport !== 'natlang' && option(args, '--agent-reader')) throw new Error('--agent-reader needs --agent-transport natlang');
  const reader: AgentReader = transport === 'natlang' ? await declareReader(option(args, '--agent-reader') ?? 'text', root, { apiKey }) :
    { kind: 'text' };
  // One natlang driver per thinking setting, sent as the template argument pi-ai's qwen-chat-template format sends.
  const drivers = new Map<boolean, ModelDriver>();
  const driver = (reasoning: boolean): ModelDriver => {
    const settings = { endpoint: root, model: modelId, apiKey,
      request: { chat_template_kwargs: { enable_thinking: reasoning, preserve_thinking: true } } };
    if (!drivers.has(reasoning)) drivers.set(reasoning, reader.kind === 'neuralese' ?
      neuraleseServerModelTurn({ ...settings, store }) : openAICompatibleModelTurn(settings));
    return drivers.get(reasoning)!;
  };
  const models = createModels();
  models.setProvider(createProvider({
    id: 'agent', name: 'Agent', baseUrl,
    auth: { apiKey: { name: 'agent key', resolve: async () => ({ auth: { apiKey: apiKey ?? 'none' } }) } },
    models: [{ id: modelId, name: modelId, api: transport === 'natlang' ? 'natlang-model-turn' : 'openai-completions', provider: 'agent',
      baseUrl, input: ['text'], reasoning: true, reader,
      contextWindow: Number(option(args, '--context-window') ?? 65536), maxTokens: Number(option(args, '--max-tokens') ?? 16384),
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      compat: { thinkingFormat: 'qwen-chat-template', supportsDeveloperRole: false, supportsStore: false, supportsReasoningEffort: false,
        maxTokensField: 'max_tokens' } } as never],
    api: transport === 'natlang' ? natlangApi((_model, { reasoning }) => driver(reasoning)) : openAICompletionsApi(),
  }));
  return { models, ref: { provider: 'agent', modelId } };
}

/** The mode a pluggable point's flag names: its value, else nl under --pure, else crisp. */
const mode = (args: string[], name: string): PluggableMode => pluggableMode(option(args, `--${name}`), args.includes('--pure') ? 'nl' : 'crisp');

function implementations(args: string[]): Implementations {
  return { context: mode(args, 'context'), scheduler: mode(args, 'scheduler'), admission: mode(args, 'admission'), planning: mode(args, 'planning') };
}

const textOf = (message: AssistantMessage | undefined) =>
  (message?.content ?? []).flatMap(item => item.type === 'text' ? [item.text] : []).join('\n').trim();

/** The launcher's runtime, with the executor's context budget when --executor-context sets it. */
function executor(target: TargetContext, args: string[]): NatlangRuntime {
  const budget = option(args, '--executor-context');
  if (budget === undefined) return target.runtime;
  const configured = target.runtime.options.model;
  const base = typeof configured === 'object' && configured ? configured : {};
  const driver = typeof configured === 'function' ? configured : (configured as { driver?: unknown } | undefined)?.driver ?? target.model;
  return new NatlangRuntime({ ...target.runtime.options,
    model: { ...base, driver, contextTokens: Number(budget) } as never });
}

export type RunResult = { status: string; answer: string; reason?: string; ms: number };

/** Run one task to its answer on a session at `sessionPath`. */
export async function runTask(target: TargetContext, args: string[], task: string, cwd: string, sessionPath: string, signal?: AbortSignal): Promise<RunResult> {
  const started = Date.now();
  const { models, ref } = await agentModels(args, (target as { modelEndpoint?: { endpoint: string; model: string } }).modelEndpoint,
    target.runtime.options.neuralese?.store);
  const quiet = args.includes('--quiet');
  const log = (line: string) => { if (!quiet) target.io.error.write(`${line}\n`); };
  const envs = createEnvs(cwd);
  const natlang = executor(target, args);
  mkdirSync(join(sessionPath, '..'), { recursive: true });
  const registry = codingRegistry(natlang, { cwd });
  let opened: Harness | undefined;
  // The companion (COMPANION.md) watches the agent's work in the background and briefs it each request.
  if (args.includes('--companion')) registry.install(companion(natlang, { harness: () => opened!,
    shaping: mode(args, 'shaping'),
    onReport: error => log(`  [companion] ${error instanceof Error ? error.message : String(error)}`) }));
  const harness = await openPi({
    storage: await openNodeSqliteStorage(sessionPath),
    natlang,
    models,
    registry,
    env: envs.env,
    implementations: implementations(args),
    onReport: error => log(`  [report] ${error instanceof Error ? error.message : String(error)}`),
    onPhase: event => log(`  ${event.kind}#${event.taskId} ${event.phase}${event.mode === 'abort' ? ' (abort)' : ''}` +
      `${event.attempt > 1 ? ` attempt ${event.attempt}` : ''}: ${event.error ? `failed: ${event.error.slice(0, 300)}` : (event.summary ?? '').slice(0, 200)}`),
  }, context);
  opened = harness;
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
  if (args[0] === 'replay') {
    const input = option(args, '--in'), output = option(args, '--out');
    if (!input || !output) throw new Error('usage: replay --in PREPARED.jsonl --out REPLAYED.jsonl [--repos DIR]');
    await replayFile(input, output, option(args, '--repos') ?? join(process.env.HOME ?? '.', 'data/harness-bench/repos'), line => target.io.error.write(`${line}\n`));
    return 0;
  }
  if (args[0] === 'surface') {
    process.stdout.write(JSON.stringify(agentSurface({ cwd: option(args, '--cwd') ?? '/workspace', companion: args.includes('--companion') })) + '\n');
    return 0;
  }
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
