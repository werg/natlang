/**
 * `natlang run applications/pi -- [-p] [--no-system-one] [--no-codemode] [--yes] [--max-turns N] [--session FILE]
 *   [--big-endpoint URL --big-model ID [--big-key-env VAR]] TASK...`: pi's coding agent on the workspace. The launcher's
 *   model runs System One; the big model drives the loop (the same model unless --big-* names another endpoint).
 *   Without TASK (and without -p) each input line is a task.
 * `natlang run applications/pi -- eval [NAME...] [--variants plain,system-one,codemode] [--out DIR]`: the tasks in
 *   tasks/, each on a fresh copy of its repository, judged by its check command.
 */
import { cpSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangRuntime, openAICompatibleModelTurn, type ModelDriver, type TargetContext } from '@natlang/node';
import { codemodeDriver, runAgent, type AgentEvent, type AgentOptions, type AgentResult, type CodemodeScripts } from './agent.js';
import { runShell } from './tools.js';

const taskDirectory = fileURLToPath(new URL('../../pi/tasks', import.meta.url));
const option = (args: string[], name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
const VALUED = ['--max-turns', '--session', '--big-endpoint', '--big-model', '--big-key-env', '--variants', '--out', '--cwd'];
const positional = (args: string[]) => args.filter((arg, i) => !arg.startsWith('-') && !VALUED.includes(args[i - 1] ?? ''));

/** The launcher's runtime with its model wrapped for codemode. */
function smallRuntime(context: TargetContext, scripts: CodemodeScripts): NatlangRuntime {
  const configured = context.runtime.options.model;
  const small = typeof configured === 'function' ? configured : configured?.driver ?? context.model;
  const model = codemodeDriver(small, scripts);
  return new NatlangRuntime({ ...context.runtime.options, model: typeof configured === 'object' && configured ? { ...configured, driver: model } : model });
}

function bigModel(context: TargetContext, args: string[]): ModelDriver {
  const endpoint = option(args, '--big-endpoint'), model = option(args, '--big-model');
  if (!endpoint) return context.model;
  if (!model) throw new Error('--big-endpoint needs --big-model');
  const keyVariable = option(args, '--big-key-env');
  return openAICompatibleModelTurn({ endpoint, model, ...keyVariable ? { apiKey: process.env[keyVariable] } : {} });
}

function printer(context: TargetContext): (event: AgentEvent) => void {
  return event => {
    if (event.kind === 'assistant') {
      if (event.text.trim()) context.io.error.write(`\n${event.text.trim()}\n`);
      for (const [name, args] of event.calls) context.io.error.write(`> ${name} ${JSON.stringify(args).slice(0, 160)}\n`);
    } else if (event.kind === 'tool') {
      context.io.error.write(`  ${event.ok ? '' : '! '}${event.text.split('\n').slice(0, 3).join(' | ').slice(0, 160)}\n`);
    } else {
      const { kind, value, confidence, action, error } = event.intervention;
      context.io.error.write(`  [system one: ${kind} ${value === undefined ? '' : JSON.stringify(value).slice(0, 80)}${confidence === undefined ? '' : ` p=${confidence.toFixed(2)}`} -> ${error ? `error: ${error}` : action}]\n`);
    }
  };
}

export default async function main(context: TargetContext): Promise<number> {
  const args = context.args;
  const scripts: CodemodeScripts = {};
  const runtime = smallRuntime(context, scripts);
  const big = bigModel(context, args);
  const maxTurns = Number(option(args, '--max-turns') ?? 60);

  if (args[0] === 'eval') return evaluate(context, args.slice(1), runtime, big, scripts, maxTurns);

  const cwd = resolve(context.workspace, option(args, '--cwd') ?? '.');
  const print = args.includes('-p') || args.includes('--print');
  const interactive = !print && (context.io.input as { isTTY?: boolean }).isTTY === true;
  const lines = createInterface({ input: context.io.input, output: context.io.error, terminal: interactive });
  const ask = (question: string) => new Promise<string>(answer => lines.question(question, answer));
  const confirm: AgentOptions['confirm'] = args.includes('--yes') ? async () => true : interactive ?
    async (command, decision) => /^y/i.test(await ask(`\nRun \`${command}\`? (judged ${decision.value}, p=${decision.confidence.toFixed(2)}) [y/N] `)) : undefined;
  const run = (task: string) => runAgent({ task, cwd, big, runtime, scripts, confirm, maxTurns,
    systemOne: !args.includes('--no-system-one'), codemode: !args.includes('--no-codemode'),
    session: option(args, '--session') ? resolve(context.workspace, option(args, '--session')!) : join(context.stateDirectory, 'sessions', `${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`),
    onEvent: printer(context) });

  const task = positional(args).join(' ').trim();
  try {
    if (task) {
      const result = await run(task);
      context.io.output.write(`${result.answer || `(stopped: ${result.stopped})`}\n`);
      return result.stopped === 'answered' ? 0 : 1;
    }
    for await (const line of lines) {
      if (!line.trim()) continue;
      const result = await run(line.trim());
      context.io.output.write(`${result.answer || `(stopped: ${result.stopped})`}\n`);
    }
    return 0;
  } finally { lines.close(); }
}

type Variant = 'plain' | 'system-one' | 'codemode';
type TaskSpec = { prompt: string, check: string };

/** Run each task on a fresh copy of its repository under each variant and judge it by its check. */
async function evaluate(context: TargetContext, args: string[], runtime: NatlangRuntime, big: ModelDriver, scripts: CodemodeScripts, maxTurns: number): Promise<number> {
  const names = positional(args);
  const variants = (option(args, '--variants') ?? 'plain,system-one').split(',') as Variant[];
  const out = resolve(context.workspace, option(args, '--out') ?? 'pi-eval-out');
  mkdirSync(out, { recursive: true });
  const tasks = readdirSync(taskDirectory).filter(name => !names.length || names.includes(name)).sort();
  const rows: Record<string, unknown>[] = [];
  for (const name of tasks) {
    const spec = JSON.parse(readFileSync(join(taskDirectory, name, 'task.json'), 'utf8')) as TaskSpec;
    for (const variant of variants) {
      const cwd = join(mkdtempSync(join(tmpdir(), `pi-${name}-`)), 'repo');
      cpSync(join(taskDirectory, name, 'repo'), cwd, { recursive: true });
      await runShell('git init -q && git add -A && git -c user.name=pi -c user.email=pi@example.com commit -qm start', cwd);
      context.io.error.write(`\n=== ${name} (${variant})\n`);
      let result: AgentResult | null = null, error: string | undefined;
      try {
        result = await runAgent({ task: spec.prompt, cwd, big, runtime, scripts, maxTurns, systemOne: variant !== 'plain',
          codemode: variant === 'codemode', session: join(out, `${name}.${variant}.jsonl`), onEvent: printer(context) });
      } catch (caught) { error = String((caught as Error)?.message ?? caught); }
      const check = await runShell(spec.check.replaceAll('{task}', join(taskDirectory, name)), cwd, 300);
      const row = { task: name, variant, passed: check.exitCode === 0, stopped: result?.stopped, turns: result?.turns, toolCalls: result?.toolCalls,
        promptTokens: result?.promptTokens, completionTokens: result?.completionTokens, ms: result?.ms, error,
        interventions: result?.interventions.map(({ kind, action, value, confidence, ms, error }) => ({ kind, action, value, confidence, ms, error })),
        check: check.output.slice(-1500) };
      rows.push(row);
      context.io.error.write(`=== ${name} (${variant}): ${row.passed ? 'PASS' : 'FAIL'} in ${row.turns} turns${error ? `: ${error}` : ''}\n`);
      writeFileSync(join(out, 'results.json'), JSON.stringify(rows, null, 2));
    }
  }
  for (const variant of variants) {
    const mine = rows.filter(row => row.variant === variant);
    context.io.output.write(`${variant}: ${mine.filter(row => row.passed).length}/${mine.length} passed, ` +
      `${mine.reduce((sum, row) => sum + Number(row.turns ?? 0), 0)} turns, ${mine.reduce((sum, row) => sum + Number(row.promptTokens ?? 0), 0)} prompt tokens\n`);
  }
  return 0;
}
