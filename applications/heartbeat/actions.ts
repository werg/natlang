// The owner-edited allowlist (actions.json) and the watch list (watch.json). The program reads them and never edits them.
// Reversibility, ownership and the auto-apply promise belong to the action, not to the model's proposal.
import type { ActionSummary, Cause, Proposal, WatchEntry } from './types.js';
import type { Host } from './host.js';

export type Stage = 'shadow' | 'advisory' | 'auto';
export type Mode = 'crisp' | 'nl' | 'shadow';
/** The kinds of value a parameter takes. */
export type ParamKind = 'watched-unit' | 'run-id' | 'message-id' | 'minutes' | 'number' | 'text' | 'path';
export type Auto = 'always' | 'promoted' | 'never';
export type ActionDef = {
  id: string,
  what: string,
  /** The command, argv fixed here; `{name}` is replaced by the parameter of that name, `{target}`, `{repo}` and `{machine}` by those values,
   * and the single element `{...cycle_argv}` by the settings' cycle command. null: the action is carried out inside the program or by a session. */
  argv: string[] | null,
  /** What goes to the command's standard input: "plan-summary", or null. */
  stdin: 'plan-summary' | null,
  /** Parameter name to kind. */
  params: Record<string, ParamKind>,
  /** What `target` names: a watched run, a watched unit, a message id, a path under the repo, or this machine. */
  target: 'run' | 'unit' | 'message' | 'path' | 'machine',
  applies_to: Cause[],
  /** A reading command that shows the state this action changes; run before and after for the receipt. */
  probe: string[] | null,
  reversible: boolean,
  process_control: boolean,
  /** Who decides: "program", "machine-session" or "owning-session". */
  owner: string,
  auto: Auto,
  /** false: never proposed as a command (a session edits and commits). */
  proposable: boolean,
};
export type Settings = {
  stage: Stage,
  modes: { diagnoseRun: Mode, triageInbox: Mode, planNext: Mode },
  /** How often a rejected answer goes back to the function with the checker's message. */
  repairs: number,
  timer_interval_minutes: number,
  executor_endpoint: string,
  max_busy: number,
  idle_wait_seconds: number,
  teacher_metrics_url: string,
  teacher_window_file: string | null,
  /** Repository root the collectors and scripts run in. null: the HEARTBEAT_REPO variable, else the current directory. */
  repo: string | null,
  data_roots: string[],
  /** Units the machine runs that need no watch entry (regular expressions on the unit name). */
  ignore_units: string[],
  inbox_hours: number,
  history_cycles: number,
  /** A resource counts as idle at or below/above these. */
  idle: { headroom_gb: number, gpu_utilization: number, teacher_load: number },
  cycle_argv: string[],
  record_dir: string,
};
export type ActionsFile = { schema: 'natlang.heartbeat-actions/1', settings: Settings, actions: ActionDef[] };
export type WatchFile = { schema: 'natlang.heartbeat-watch/1', runs: WatchEntry[] };

/** Process control is a proposal to the owning session in every stage, whatever a file or a plan says. */
export const PROCESS_CONTROL = new Set(['relaunch-run', 'stop-unit', 'adopt-unit', 'queue-next']);
const STAGES: Stage[] = ['shadow', 'advisory', 'auto'];
const RANK: Record<Stage, number> = { shadow: 0, advisory: 1, auto: 2 };
export const MODES: Mode[] = ['crisp', 'nl', 'shadow'];

/** The lower of two stages: a command-line stage can hold a cycle back, never promote it. */
export function lowerStage(file: Stage, requested: Stage | undefined): Stage {
  return requested && RANK[requested] < RANK[file] ? requested : file;
}

/** Whether the applier may carry this action out by itself at this stage. */
export function autoApplies(def: ActionDef, stage: Stage): boolean {
  if (!def.reversible || def.process_control || PROCESS_CONTROL.has(def.id)) return false;
  if (def.auto === 'always') return true;
  return def.auto === 'promoted' && stage === 'auto';
}

export type Loaded<T> = { value: T, warnings: string[] };

export async function loadActions(host: Host, path: string): Promise<Loaded<ActionsFile>> {
  const file = await host.readText(path);
  if (!file) throw new Error(`the allowlist ${path} does not exist; the owner provides it (applications/heartbeat/actions.json)`);
  const parsed = JSON.parse(file.text) as ActionsFile;
  if (parsed.schema !== 'natlang.heartbeat-actions/1') throw new Error(`${path}: schema must be "natlang.heartbeat-actions/1"`);
  const warnings: string[] = [];
  if (!STAGES.includes(parsed.settings.stage)) throw new Error(`${path}: settings.stage is one of ${STAGES.join(', ')}`);
  for (const [name, mode] of Object.entries(parsed.settings.modes)) if (!MODES.includes(mode)) throw new Error(`${path}: settings.modes.${name} is one of ${MODES.join(', ')}`);
  const ids = new Set<string>();
  for (const def of parsed.actions) {
    if (ids.has(def.id)) throw new Error(`${path}: action id ${def.id} appears twice`);
    ids.add(def.id);
    // A file cannot promise automatic application for what cannot be undone; the program holds the action back and says so.
    if (def.auto !== 'never' && (!def.reversible || def.process_control || PROCESS_CONTROL.has(def.id))) {
      warnings.push(`${def.id} is marked auto "${def.auto}" but is ${def.process_control || PROCESS_CONTROL.has(def.id) ? 'process control' : 'not reversible'}; it stays a proposal`);
      def.auto = 'never';
    }
  }
  return { value: parsed, warnings };
}

export async function loadWatch(host: Host, path: string, machine: string): Promise<WatchEntry[]> {
  const file = await host.readText(path);
  if (!file) return [];
  const parsed = JSON.parse(file.text) as WatchFile;
  if (parsed.schema !== 'natlang.heartbeat-watch/1') throw new Error(`${path}: schema must be "natlang.heartbeat-watch/1"`);
  return parsed.runs.filter(entry => entry.machine === machine);
}

/** What the planning function is shown of an action. */
export function summarize(def: ActionDef): ActionSummary {
  return { id: def.id, what: def.what, params: { ...def.params }, applies_to: def.applies_to };
}

export type Known = { runs: string[], units: string[], messages: string[], machine: string, timerIntervalMinutes: number, repo: string };

const oneLine = (text: string) => !/[\r\n]/.test(text);
/** Problems of a proposal measured against the allowlist and what exists on the machine: the exact verifier of a plan. */
export function proposalProblems(proposal: Proposal, defs: ActionDef[], known: Known): string[] {
  const def = defs.find(item => item.id === proposal.action);
  if (!def) return [`action "${proposal.action}" is not in the allowlist; the ids are ${defs.filter(item => item.proposable).map(item => item.id).join(', ')}`];
  if (!def.proposable) return [`action "${def.id}" is carried out by a session, not proposed as a command`];
  const problems: string[] = [];
  const names = (list: string[]) => list.length ? list.join(', ') : 'none';
  switch (def.target) {
    case 'run': if (!known.runs.includes(proposal.target)) problems.push(`target "${proposal.target}" of ${def.id} is a watched run id (${names(known.runs)})`); break;
    case 'unit': if (!known.units.includes(proposal.target)) problems.push(`target "${proposal.target}" of ${def.id} is a watched unit (${names(known.units)})`); break;
    case 'message': if (!known.messages.includes(proposal.target)) problems.push(`target "${proposal.target}" of ${def.id} is a message id (${names(known.messages)})`); break;
    case 'machine': if (proposal.target !== known.machine) problems.push(`target of ${def.id} is this machine, "${known.machine}"`); break;
    case 'path': if (!safePath(proposal.target)) problems.push(`target "${proposal.target}" of ${def.id} is a relative path inside the repository`); break;
  }
  const given = Object.keys(proposal.params ?? {});
  for (const name of Object.keys(def.params)) if (!given.includes(name)) problems.push(`${def.id} takes the parameter ${name} (${def.params[name]})`);
  for (const name of given) if (!(name in def.params)) problems.push(`${def.id} takes ${Object.keys(def.params).join(', ') || 'no parameters'}, not ${name}`);
  for (const [name, kind] of Object.entries(def.params)) {
    const value = proposal.params?.[name];
    if (value === undefined) continue;
    const text = String(value);
    switch (kind) {
      case 'watched-unit': if (!known.units.includes(text)) problems.push(`parameter ${name} of ${def.id} is a watched unit (${names(known.units)})`); break;
      case 'run-id': if (!known.runs.includes(text)) problems.push(`parameter ${name} of ${def.id} is a watched run id (${names(known.runs)})`); break;
      case 'message-id': if (!known.messages.includes(text)) problems.push(`parameter ${name} of ${def.id} is a message id (${names(known.messages)})`); break;
      case 'minutes': if (!/^[1-9]\d*$/.test(text) || Number(text) >= known.timerIntervalMinutes) problems.push(`parameter ${name} of ${def.id} is a positive whole number of minutes below ${known.timerIntervalMinutes}`); break;
      case 'number': if (!/^\d+(\.\d+)?$/.test(text)) problems.push(`parameter ${name} of ${def.id} is a non-negative number`); break;
      case 'text': if (!text.trim() || text.length > 2000) problems.push(`parameter ${name} of ${def.id} is text of 1 to 2000 characters`); break;
      case 'path': if (!safePath(text)) problems.push(`parameter ${name} of ${def.id} is a relative path inside the repository`); break;
    }
  }
  return problems;
}

function safePath(path: string): boolean {
  return !!path && !path.startsWith('/') && !path.split('/').includes('..') && oneLine(path);
}

/** The command an action stands for with its parameters in place; null when the action has no command. */
export function commandFor(def: ActionDef, proposal: Proposal, known: Pick<Known, 'repo' | 'machine'>, cycleArgv: string[]): string[] | null {
  if (!def.argv) return null;
  const fill = (part: string) => part.replace(/\{(\w+)\}/g, (_, name: string) =>
    name === 'target' ? proposal.target : name === 'repo' ? known.repo : name === 'machine' ? known.machine : String(proposal.params[name] ?? ''));
  return def.argv.flatMap(part => part === '{...cycle_argv}' ? cycleArgv.map(fill) : [fill(part)]);
}
