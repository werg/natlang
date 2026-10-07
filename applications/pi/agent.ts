/**
 * pi's agent loop in TypeScript: the optimized harness (pi.nl is pi as a natlang program). The big model works as in pi: pi's system prompt,
 * pi's four tools, one conversation. Small-model natural-language functions ("System One") work around it:
 * - risk gates each shell command before it runs (refuse, ask the user, or run, by probability floors);
 * - digest condenses long command output to what the current step needs (the full output stays on disk);
 * - review checks each edit's diff against what the model said it would do;
 * - progress notices when recent actions go in circles and adds a steering note;
 * - done checks a final answer against the task and the last results, and sends it back once if unfinished;
 * - scout suggests the files to start from; compact writes pi's checkpoint summary when the context fills up;
 * - route (off by default) hands a turn to the small model when the next step is routine.
 * codemode lets the big model hand natlang a script: shell and file work plus typed `nl` judgments on the small
 * model, of which the big model sees only the result.
 */
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Decision, ModelDriver, NatlangRuntime } from '@natlang/node';
import pi from './pi.nl';
import compact from './harness/compact.nl';
import route from './harness/route.nl';
import codemode from './harness/codemode.nl';
import { CODEMODE_DEFINITION, SCRIPT_DECLARATIONS, TOOL_DEFINITIONS, projectFiles, runShell, runTool, scriptServices, truncate } from './tools.js';
import type { Risk } from './types.js';

// The judgments pi.nl's tools use, run here by the optimized harness itself.
const { bash: { risk, digest }, edit: { review }, context: { scout }, progress, done } = pi;

type ToolCall = { id: string, type: 'function', function: { name: string, arguments: string } };
type Message = { role: 'system' | 'user' | 'assistant' | 'tool', content: string, tool_calls?: ToolCall[], tool_call_id?: string, reasoning?: string };

export type SystemOneOptions = {
  risk: boolean, digest: boolean, review: boolean, progress: boolean, done: boolean, scout: boolean, compact: boolean,
  /** Let the small model take routine turns (needs AgentOptions.small). */
  route: boolean,
  /** p(routine) a turn needs to go to the small model. */
  routeFloor: number,
  /** p(destructive) at or above which a command is refused. */
  deny: number,
  /** p(review) + p(destructive) at or above which the user is asked first. */
  ask: number,
  /** Confidence a steering note (progress, review, done) needs. */
  steer: number,
  /** Output longer than this many lines is digested. */
  digestLines: number,
  /** Tool calls between progress checks. */
  progressEvery: number,
  /** Times a final answer may be sent back as unfinished. */
  pushbacks: number,
};
export const SYSTEM_ONE: SystemOneOptions = { risk: true, digest: true, review: true, progress: true, done: true, scout: true,
  compact: true, route: false, routeFloor: 0.8, deny: 0.7, ask: 0.5, steer: 0.6, digestLines: 80, progressEvery: 6, pushbacks: 1 };

export type Intervention = { kind: 'risk' | 'digest' | 'review' | 'progress' | 'done' | 'scout' | 'compact' | 'route', turn: number,
  value?: unknown, confidence?: number, action: string, ms: number, error?: string };
export type AgentEvent = { kind: 'assistant', turn: number, text: string, calls: [string, Record<string, unknown>][] } |
  { kind: 'tool', turn: number, name: string, ok: boolean, text: string } | { kind: 'system-one', intervention: Intervention };

export type AgentOptions = {
  task: string,
  cwd: string,
  /** Drives the loop. */
  big: ModelDriver,
  /** Takes the turns route judges routine. */
  small?: ModelDriver,
  /** Runs System One and codemode scripts; its model must be wrapped with codemodeDriver. */
  runtime: NatlangRuntime,
  scripts: CodemodeScripts,
  systemOne?: boolean | Partial<SystemOneOptions>,
  codemode?: boolean,
  /** Asked before a command judged to need review runs; without it such commands are refused. */
  confirm?: (command: string, decision: Decision<Risk>) => Promise<boolean>,
  maxTurns?: number,
  maxTokens?: number,
  temperature?: number,
  /** Context size (characters of the request) at which old turns are compacted, and how much recent history stays. */
  contextChars?: number,
  keepChars?: number,
  /** Session log (JSONL). */
  session?: string,
  /** Skill directories beyond pi's own (~/.pi/agent/skills, <cwd>/.pi/skills). */
  skillDirs?: string[],
  onEvent?: (event: AgentEvent) => void,
  signal?: AbortSignal,
};
export type AgentResult = { answer: string, stopped: 'answered' | 'max-turns' | 'aborted' | 'failed', turns: number, smallTurns: number, toolCalls: number,
  interventions: Intervention[], promptTokens: number, completionTokens: number, ms: number, error?: string };

const MARKER = 'Run the script you were given in eval, exactly as written';
export type CodemodeScripts = { current?: string };

/**
 * The small model, except for codemode calls: the call running the agent's script is answered by evaluating the
 * script and returning what it observed, while the `nl` calls the script makes reach the small model.
 */
export function codemodeDriver(small: ModelDriver, scripts: CodemodeScripts): ModelDriver {
  const driver: ModelDriver = async (request, signal) => {
    const messages = request.messages as Message[];
    if (scripts.current === undefined || !messages.some(message => JSON.stringify(message.content ?? '').includes(MARKER)))
      return small(request, signal);
    const calls = messages.flatMap(message => message.role === 'assistant' ?
      (message.tool_calls ?? []).filter(call => !call.id.startsWith('scope_')) : []);
    const evaluation = calls.find(call => call.function.name === 'eval');
    if (!evaluation) return { calls: [['eval', { code: scripts.current }]] };
    const observed = messages.find(message => message.role === 'tool' && message.tool_call_id === evaluation.id)?.content ?? '';
    const retried = calls.some(call => call.function.name === 'return_result');
    return { calls: [['return_result', retried ? { status: 'failed', reason: String(observed) } : { status: 'success', value: String(observed) }]] };
  };
  return Object.assign(driver, small);
}

export type Skill = { name: string, description: string, path: string };

/**
 * pi's skills: a directory holding SKILL.md (frontmatter name and description) is one skill, found under
 * ~/.pi/agent/skills, <cwd>/.pi/skills and any extra directories. The model reads a skill's file when the task
 * matches its description; `disable-model-invocation: true` hides it.
 */
export function discoverSkills(cwd: string, extra: string[] = [], home = homedir()): Skill[] {
  const found = new Map<string, Skill>();
  const visit = (dir: string, depth: number) => {
    if (depth > 4 || !existsSync(dir) || !statSync(dir).isDirectory()) return;
    const file = join(dir, 'SKILL.md');
    if (existsSync(file)) {
      const head = /^---\n([^]*?)\n---/.exec(readFileSync(file, 'utf8'))?.[1] ?? '';
      const field = (key: string) => new RegExp(`^${key}:\\s*(.*)$`, 'm').exec(head)?.[1]?.trim().replace(/^(['"])(.*)\1$/, '$2');
      const description = field('description');
      if (description && field('disable-model-invocation') !== 'true' && !found.has(file))
        found.set(file, { name: field('name') || basename(dir), description, path: file });
      return;
    }
    for (const entry of readdirSync(dir).sort()) if (!entry.startsWith('.') && entry !== 'node_modules') visit(join(dir, entry), depth + 1);
  };
  for (const dir of [join(home, '.pi', 'agent', 'skills'), join(resolve(cwd), '.pi', 'skills'), ...extra]) visit(resolve(dir), 0);
  return [...found.values()];
}

const xml = (text: string) => text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/** pi's system prompt: preamble, tools, rules, project instructions, skills, working directory. */
export function systemPrompt(cwd: string, options: { codemode: boolean, systemOne: boolean, skills?: Skill[] }): string {
  const tools = [['read', 'Read file contents'], ['bash', 'Execute bash commands (ls, grep, find, etc.)'],
    ['edit', 'Make precise file edits with exact text replacement, including multiple disjoint edits in one call'],
    ['write', 'Create or overwrite files'],
    ...options.codemode ? [['codemode', 'Run a script of shell, file and small-model steps and see only its result']] : []];
  const rules = ['Use bash for file operations like ls, rg, find', 'Use read to examine files instead of cat or sed.',
    'Use edit for precise changes (edits[].oldText must match exactly)',
    'When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls',
    'Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.',
    'Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.',
    'Use write only for new files or complete rewrites.',
    ...options.codemode ? ['Use codemode when a step takes many commands or a judgement over many items (every call site, every failing test): the script does them and you read only what it reports.'] : [],
    ...options.systemOne ? ['Notes in [square brackets] in tool results come from a smaller, faster model watching your work: digests of long output (the full output is saved to the file they name), reviews of edits, progress notes, refusals of risky commands. They are quick judgements; weigh them and check when it matters.'] : [],
    'Be concise in your responses', 'Show file paths clearly when working with files'];
  const sections: [string, string][] = [
    ['tools', `${tools.map(([name, text]) => `- ${name}: ${text}`).join('\n')}\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.`],
    ['rules', rules.map(rule => `- ${rule}`).join('\n')]];
  const context = contextFiles(cwd);
  if (context.length) sections.push(['project_context', ['Project-specific instructions and guidelines:', ...context.map(({ path, content }) =>
    `<project_instructions path="${path}">\n${content}\n</project_instructions>`)].join('\n\n')]);
  if (options.skills?.length) sections.push(['skills', ['The following skills provide specialized instructions for specific tasks.',
    "Use the read tool to load a skill's file when the task matches its description.",
    "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
    '', '<available_skills>', ...options.skills.flatMap(skill => ['  <skill>', `    <name>${xml(skill.name)}</name>`,
      `    <description>${xml(skill.description)}</description>`, `    <location>${xml(skill.path)}</location>`, '  </skill>']),
    '</available_skills>'].join('\n')]);
  sections.push(['cwd', cwd]);
  return ['You are an expert coding assistant operating inside pi, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.',
    ...sections.map(([name, content]) => `<${name}>\n${content}\n</${name}>`)].join('\n\n');
}

/** AGENTS.md (or CLAUDE.md) from the working directory and its ancestors, outermost first. */
function contextFiles(cwd: string): { path: string, content: string }[] {
  const found: { path: string, content: string }[] = [];
  for (let dir = resolve(cwd); ; dir = dirname(dir)) {
    const file = ['AGENTS.md', 'CLAUDE.md'].map(name => join(dir, name)).find(path => existsSync(path));
    if (file) found.unshift({ path: file, content: readFileSync(file, 'utf8') });
    if (dirname(dir) === dir) return found;
  }
}

/** pi's serialization of a conversation for summarizing. */
function serialize(messages: Message[]): string {
  return messages.flatMap(message => {
    if (message.role === 'user') return message.content ? [`[User]: ${message.content}`] : [];
    if (message.role === 'tool') return [`[Tool result]: ${message.content.length > 2000 ? `${message.content.slice(0, 2000)}\n\n[... ${message.content.length - 2000} more characters truncated]` : message.content}`];
    if (message.role !== 'assistant') return [];
    const calls = (message.tool_calls ?? []).map(call => `${call.function.name}(${Object.entries(JSON.parse(call.function.arguments) as Record<string, unknown>)
      .map(([key, value]) => `${key}=${JSON.stringify(value)}`).join(', ')})`);
    return [...message.reasoning ? [`[Assistant thinking]: ${message.reasoning}`] : [], ...message.content ? [`[Assistant]: ${message.content}`] : [],
      ...calls.length ? [`[Assistant tool calls]: ${calls.join('; ')}`] : []];
  }).join('\n\n');
}

const brief = (args: Record<string, unknown>) => Object.entries(args).map(([key, value]) =>
  `${key}=${JSON.stringify(value).slice(0, key === 'command' || key === 'path' ? 300 : 80)}`).join(', ');

export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const started = performance.now();
  const one: SystemOneOptions | null = options.systemOne === false ? null :
    { ...SYSTEM_ONE, ...typeof options.systemOne === 'object' ? options.systemOne : {} };
  const { runtime, cwd, task } = options;
  const useCodemode = options.codemode ?? true;
  const tools = [...TOOL_DEFINITIONS, ...useCodemode ? [CODEMODE_DEFINITION] : []];
  const contextChars = options.contextChars ?? 160_000, keepChars = options.keepChars ?? 40_000;
  const interventions: Intervention[] = [];
  const actions: string[] = [];
  let turn = 0, smallTurns = 0, toolCalls = 0, promptTokens = 0, completionTokens = 0, pushbacks = 0, lastProgressCheck = 0;

  // Session log: a header, then every message and System One step as it happens.
  let parentId: string | null = null;
  const log = (entry: Record<string, unknown>) => {
    if (!options.session) return;
    const id = randomUUID().slice(0, 8);
    appendFileSync(options.session, JSON.stringify({ ...entry, id, parentId, timestamp: new Date().toISOString() }) + '\n');
    parentId = id;
  };
  if (options.session) {
    mkdirSync(dirname(options.session), { recursive: true });
    writeFileSync(options.session, JSON.stringify({ type: 'session', version: 3, id: randomUUID(), timestamp: new Date().toISOString(), cwd }) + '\n');
  }

  // A System One step: timed, logged, and never fatal to the agent.
  async function step<T>(kind: Intervention['kind'], work: () => Promise<{ value?: unknown, confidence?: number, action: string, result: T }>, fallback: T): Promise<T> {
    const begun = performance.now();
    let intervention: Intervention, result = fallback;
    try {
      const outcome = await work();
      result = outcome.result;
      intervention = { kind, turn, value: outcome.value, confidence: outcome.confidence, action: outcome.action, ms: Math.round(performance.now() - begun) };
    } catch (error) {
      intervention = { kind, turn, action: 'skipped', ms: Math.round(performance.now() - begun), error: String((error as Error)?.message ?? error).slice(0, 500) };
    }
    interventions.push(intervention);
    log({ type: 'system_one', ...intervention });
    options.onEvent?.({ kind: 'system-one', intervention });
    return result;
  }
  const probability = <T>(decision: Decision<T>, value: T) => decision.probabilities.find(item => item.value === value)?.probability ?? 0;
  const confident = (decision: Decision<string>) => decision.confidence >= one!.steer;

  /** Why a command may not run (null: it may). */
  async function gate(command: string): Promise<string | null> {
    if (!one?.risk) return null;
    return step('risk', async () => {
      const decision = await runtime.decide(risk, command, task);
      const destructive = probability(decision, 'destructive'), reviewed = destructive + probability(decision, 'review');
      const odds = `p(destructive)=${destructive.toFixed(2)}, p(review or worse)=${reviewed.toFixed(2)}`;
      if (destructive >= one.deny)
        return { value: decision.value, confidence: destructive, action: 'refused', result: `[Refused by the safety check: this command looks destructive (${odds}). Find a safer way to do it, or ask the user to run it.]` };
      if (reviewed >= one.ask) {
        const approved = options.confirm ? await options.confirm(command, decision) : false;
        return { value: decision.value, confidence: reviewed, action: approved ? 'approved by the user' : 'refused: needs approval',
          result: approved ? null : `[Not run: this command needs the user's approval (${odds}), and none was given. Do the task without it if you can; otherwise say in your answer which command the user should run and why.]` };
      }
      return { value: decision.value, confidence: decision.confidence, action: 'allowed', result: null };
    }, null);
  }

  const services = scriptServices(cwd, gate);
  async function runScript(script: string): Promise<{ ok: boolean, text: string }> {
    options.scripts.current = script;
    try {
      const observed = await runtime.run(() => codemode(script), { services, serviceDeclarations: SCRIPT_DECLARATIONS, signal: options.signal });
      return { ok: true, text: truncate(String(observed), 'tail').text };
    } catch (error) {
      return { ok: false, text: `Script failed: ${String((error as Error)?.message ?? error).slice(0, 4000)}` };
    } finally { options.scripts.current = undefined; }
  }

  // The opening: pi's prompt and the task, with the scout's suggestions.
  let opening = task;
  if (one?.scout) opening = await step('scout', async () => {
    const files = await projectFiles(cwd);
    if (files.length < 12) return { action: 'too few files to scout', result: task };
    const picked = (await runtime.run(() => scout(task, files))).filter(file => files.includes(file)).slice(0, 8);
    return { value: picked, action: picked.length ? 'suggested files' : 'nothing suggested',
      result: picked.length ? `${task}\n\n[Files a quick scan suggests starting from (unverified): ${picked.join(', ')}]` : task };
  }, task);
  const system: Message = { role: 'system', content: systemPrompt(cwd, { codemode: useCodemode, systemOne: one !== null,
    skills: discoverSkills(cwd, options.skillDirs) }) };
  const first: Message = { role: 'user', content: opening };
  log({ type: 'message', message: first });
  let summary = '';
  let history: Message[] = [];
  const messages = (): Message[] => [system, first, ...summary ? [{ role: 'user' as const,
    content: `The conversation history before this point was compacted into the following summary:\n\n<summary>\n${summary}\n</summary>` }] : [], ...history];
  const push = (message: Message) => { history.push(message); log({ type: 'message', message }); };

  async function compactHistory() {
    let cut = history.length, size = 0;
    for (let i = history.length - 1; i > 0; i--) {
      size += JSON.stringify(history[i]).length;
      if (size > keepChars) break;
      if (history[i]!.role !== 'tool') cut = i;
    }
    if (cut <= 1 || cut >= history.length) return;
    const before = JSON.stringify(messages()).length;
    const next = await step('compact', async () => {
      const written = await runtime.run(() => compact(summary, serialize(history.slice(0, cut))));
      return { action: `compacted ${cut} messages`, result: written };
    }, null as string | null);
    if (next === null) return;
    summary = next;
    history = history.slice(cut);
    log({ type: 'compaction', summary, charsBefore: before });
  }

  const maxTurns = options.maxTurns ?? 60;
  while (turn < maxTurns) {
    if (options.signal?.aborted) return finish('', 'aborted');
    turn++;
    if (one?.compact && JSON.stringify(messages()).length > contextChars) await compactHistory();
    // A routine next step can go to the small model; anything else, or any doubt, stays with the big one.
    let driver = options.big;
    if (one?.route && options.small && actions.length) driver = await step('route', async () => {
      const decision = await runtime.decide(route, task, actions.slice(-4));
      const small = decision.value === 'routine' && probability(decision, 'routine') >= one.routeFloor;
      return { value: decision.value, confidence: decision.confidence, action: small ? 'small model' : 'big model', result: small ? options.small! : options.big };
    }, options.big);
    if (driver !== options.big) smallTurns++;
    let reply: Awaited<ReturnType<ModelDriver>>;
    try {
      reply = await driver({ messages: messages(), tools, seed: null, max_tokens: options.maxTokens ?? 16384,
        ...options.temperature === undefined ? {} : { temperature: options.temperature } }, options.signal);
    } catch (error) {
      // Stopped from outside (a time limit): what the run did so far is its result.
      if (options.signal?.aborted) return finish('', 'aborted');
      throw error;
    }
    promptTokens += reply.prompt_tokens ?? 0;
    completionTokens += reply.completion_tokens ?? 0;
    const text = typeof reply.text === 'string' ? reply.text : '';
    const calls = reply.calls ?? [];
    options.onEvent?.({ kind: 'assistant', turn, text, calls });
    const toolCallsOut = calls.map(([name, args], index): ToolCall => ({ id: `pi_${turn}_${index}`, type: 'function', function: { name, arguments: JSON.stringify(args) } }));
    push({ role: 'assistant', content: text, ...calls.length ? { tool_calls: toolCallsOut } : {}, ...reply.reasoning ? { reasoning: reply.reasoning } : {} });

    if (!calls.length) {
      // A final answer. System One checks it against the task once before it stands.
      if (one?.done && pushbacks < one.pushbacks) {
        const note = await step('done', async () => {
          const decision = await runtime.decide(done, task, text, actions.slice(-10).join('\n') || '(no actions)');
          const sendBack = decision.value === 'unfinished' && confident(decision);
          return { value: decision.value, confidence: decision.confidence, action: sendBack ? 'sent back' : 'accepted',
            result: sendBack ? `[A quick check judged this unfinished (p=${decision.confidence.toFixed(2)}). Re-read the task and your last results: finish what is missing, or say briefly why it is complete.]` : null };
        }, null as string | null);
        if (note) { pushbacks++; push({ role: 'user', content: note }); continue; }
      }
      return finish(text, 'answered');
    }

    const intent = (text || reply.reasoning || '').slice(-1500);
    for (const [index, [name, args]] of calls.entries()) {
      toolCalls++;
      let outcome: { ok: boolean, text: string };
      if (name === 'bash') {
        const refusal = await gate(String(args.command));
        outcome = refusal ? { ok: false, text: refusal } : await runTool(name, args, cwd, options.signal);
        const full = (outcome as { full?: string }).full;
        const lineCount = full ? full.split('\n').length - (full.endsWith('\n') ? 1 : 0) : 0;
        if (!refusal && one?.digest && full && lineCount > one.digestLines) {
          const condensed = await step('digest', async () => {
            const file = join(mkdtempSync(join(tmpdir(), 'pi-bash-')), 'output.txt');
            writeFileSync(file, full);
            const written = await runtime.run(() => digest(truncate(full, 'tail').text, intent || task));
            const status = outcome.text.match(/\n\[(Exit code \d+|Timed out[^\]]*)\]$/)?.[0] ?? '';
            return { action: `digested ${lineCount} lines`, result: `${written}\n[A small model condensed ${lineCount} lines of output; the full output is in ${file}]${status}` };
          }, null as string | null);
          if (condensed) outcome = { ok: outcome.ok, text: condensed };
        }
      } else if ((name === 'edit' || name === 'write') && one?.review && intent) {
        const path = resolve(cwd, String(args.path));
        const before = existsSync(path) ? readFileSync(path, 'utf8') : '';
        outcome = await runTool(name, args, cwd, options.signal);
        if (outcome.ok) {
          const note = await step('review', async () => {
            const scratch = mkdtempSync(join(tmpdir(), 'pi-edit-'));
            writeFileSync(join(scratch, 'before'), before);
            const diff = (await runShell(`diff -u --label a/${String(args.path)} --label b/${String(args.path)} before ${JSON.stringify(path)}`, scratch)).output;
            const decision = await runtime.decide(review, intent, truncate(diff, 'head').text);
            const flag = decision.value !== 'as-intended' && confident(decision);
            return { value: decision.value, confidence: decision.confidence, action: flag ? 'flagged' : 'passed',
              result: flag ? `\n[Edit review: this change looks ${decision.value === 'unintended' ? 'broader than what you said you would do' : 'incomplete for what you said you would do'} (p=${decision.confidence.toFixed(2)}). Check the diff.]` : '' };
          }, '');
          outcome = { ...outcome, text: outcome.text + note };
        }
      } else if (name === 'codemode' && useCodemode) {
        outcome = await runScript(String(args.script ?? ''));
      } else {
        outcome = await runTool(name, args, cwd, options.signal);
      }
      actions.push(`${name}(${brief(args)}) -> ${outcome.ok ? '' : 'FAILED: '}${outcome.text.slice(0, 400)}`);

      // Every few tool calls System One looks at the recent actions.
      if (one?.progress && index === calls.length - 1 && toolCalls - lastProgressCheck >= one.progressEvery) {
        lastProgressCheck = toolCalls;
        outcome.text += await step('progress', async () => {
          const decision = await runtime.decide(progress, task, actions.slice(-one.progressEvery));
          const steer = decision.value !== 'progressing' && confident(decision);
          return { value: decision.value, confidence: decision.confidence, action: steer ? 'steered' : 'quiet',
            result: steer ? `\n[Progress check: your recent actions look ${decision.value === 'repeating' ? 'repetitive' : 'stuck'} (p=${decision.confidence.toFixed(2)}). Step back: re-read the last error, question your assumption, and try a different approach.]` : '' };
        }, '');
      }
      options.onEvent?.({ kind: 'tool', turn, name, ok: outcome.ok, text: outcome.text });
      push({ role: 'tool', tool_call_id: toolCallsOut[index]!.id, content: outcome.text });
    }
  }
  return finish('', 'max-turns');

  function finish(answer: string, stopped: AgentResult['stopped']): AgentResult {
    const result: AgentResult = { answer, stopped, turns: turn, smallTurns, toolCalls, interventions, promptTokens, completionTokens, ms: Math.round(performance.now() - started) };
    log({ type: 'result', ...result, interventions: interventions.length });
    return result;
  }
}
