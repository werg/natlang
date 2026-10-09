// The crisp side of the three decisions: a status ladder over unit state, progress age and gate fields; a rule table over
// message kinds; an allowlist lookup by cause. It is the `crisp` mode of each part, the reference a shadow run compares
// the natural-language side with, and the whole program when no executor answers.
import type { Untrusted } from '@natlang/node';
import { gateFailures } from './collect/gate.js';
import type { ActionDef, Settings } from './actions.js';
import type { Cause, CoordMessage, CycleMemory, Health, InboxTriage, Obligation, Plan, Proposal, Resources, RunDiagnosis, RunEvidence, ActionSummary, WatchedRun } from './types.js';

const clip = (text: string, length = 300) => text.length > length ? text.slice(0, length) : text;
const firstLine = (text: string) => clip(text.split('\n').find(line => line.trim() !== '') ?? text.trim());
const sentence = (text: string) => text.replace(/[.!?]+(\s|$)/g, ' ').replace(/\s+/g, ' ').trim();

type Rule = { cause: Cause, pattern: RegExp };
// The first rule that matches the last explaining line wins.
const CAUSES: Rule[] = [
  { cause: 'out-of-memory', pattern: /\bKilled\b|\bOOM\b|out of memory|MemoryError/i },
  { cause: 'external', pattern: /connection refused|timed out|unreachable|temporary failure in name resolution/i },
  { cause: 'admission-refused', pattern: /refus|admission|no room|reserve/i },
  { cause: 'input-missing', pattern: /No such file|FileNotFoundError|does not exist|not found|missing/i },
  { cause: 'code-error', pattern: /Traceback|Error\b|Exception/ },
];

function readingOf(run: RunEvidence, source: string) {
  return run.readings.find(reading => reading.source === source);
}

/** A quote that occurs in the reading: a named line when it has one, else its first line. */
function quoteFrom(text: string, preferred: string[]): string {
  for (const needle of preferred) if (needle && text.includes(needle)) return needle;
  return firstLine(text);
}

export function crispDiagnose(run: RunEvidence): RunDiagnosis {
  const id = run.entry.run_id, unit = run.unit;
  const evidence: RunDiagnosis['evidence'] = [];
  const cite = (source: string, preferred: string[] = []) => {
    const reading = readingOf(run, source);
    if (reading && !evidence.some(item => item.reading_id === reading.id)) evidence.push({ reading_id: reading.id, quote: quoteFrom(reading.text, preferred) });
  };
  let health: Health, cause: Cause = 'none', confidence: RunDiagnosis['confidence'] = 'medium';
  const failedGates = gateFailures(run.gates);
  const stopped = unit.active_state === 'failed' || (unit.active_state === 'inactive' || unit.active_state === 'dead') && (unit.exit_status ?? 0) !== 0 || unit.result !== 'success' && unit.result !== 'unknown' && unit.active_state !== 'active' && unit.active_state !== 'activating';
  const exitedClean = (unit.active_state === 'inactive' || unit.active_state === 'dead') && !stopped;
  if (failedGates.length) {
    health = 'gate-failed'; cause = 'gate-failed'; confidence = 'high';
    const gate = run.readings.find(reading => reading.source === 'gate' && reading.text.includes(`"${failedGates[0]!.field}"`));
    if (gate) evidence.push({ reading_id: gate.id, quote: `"${failedGates[0]!.field}": ${JSON.stringify(failedGates[0]!.value)}` });
  } else if (unit.active_state === 'unknown') {
    health = 'unknown'; cause = 'unknown'; confidence = 'low'; cite('unit');
  } else if (stopped) {
    health = 'finished-failed'; cite('unit', [`ActiveState=${unit.active_state}`, `Result=${unit.result}`]);
  } else if (exitedClean) {
    health = run.entry.expected === 'finishes' ? 'finished-ok' : 'blocked'; cite('unit', [`ActiveState=${unit.active_state}`]);
  } else if (run.entry.expected === 'serves') {
    health = 'progressing'; cite('unit', [`ActiveState=${unit.active_state}`]);
  } else if (run.minutes_since_progress === null) {
    health = 'unknown'; cause = 'unknown'; confidence = 'low'; cite('unit', [`ActiveState=${unit.active_state}`]); cite('log');
  } else if (run.minutes_since_progress >= run.entry.stall_after_minutes) {
    health = 'stalled'; cause = 'stalled-no-error'; cite('log');
  } else {
    health = 'progressing'; cite('log'); if (!evidence.length) cite('unit', [`ActiveState=${unit.active_state}`]);
  }
  if (health === 'finished-failed' || health === 'stalled' || health === 'blocked') {
    const explaining = [...run.error_candidates].reverse().map(candidate => ({ candidate, rule: CAUSES.find(rule => rule.pattern.test(candidate.line)) })).find(item => item.rule);
    const claim = run.ledger_claim;
    if (explaining) {
      cause = explaining.rule!.cause; confidence = 'high';
      evidence.unshift({ reading_id: explaining.candidate.reading_id, quote: explaining.candidate.line });
    } else if (claim && claim.budget_gb > 0 && claim.used_gb >= 0.9 * claim.budget_gb && run.headroom_gb !== null && run.headroom_gb <= 10) {
      cause = 'resource-contention'; confidence = 'medium'; cite('ledger');
    } else if (health === 'finished-failed') { cause = 'unknown'; confidence = 'low'; }
  }
  const kept = evidence.slice(0, 3);
  return { run_id: id, health, cause, evidence: kept, summary: `${id} is ${health} (cause: ${cause}).`, confidence };
}

const URGENT_FIRST = (a: Obligation, b: Obligation, byId: Map<string, CoordMessage>) =>
  Number(byId.get(b.message_id)?.urgent) - Number(byId.get(a.message_id)?.urgent) || ['answer-request', 'adopt-decision', 'acknowledge', 'inform'].indexOf(a.kind) - ['answer-request', 'adopt-decision', 'acknowledge', 'inform'].indexOf(b.kind);

/** Reads message kinds; whether a decision changes how this machine runs something is a judgment, so decisions are `adopt-decision` here. */
export function crispTriage(messages: CoordMessage[], machine: string, watched: WatchedRun[]): InboxTriage {
  void machine;
  const obligations: Obligation[] = messages.map(message => {
    const kind: Obligation['kind'] = message.kind === 'request' ? 'answer-request' : message.kind === 'decision' ? 'adopt-decision' : 'inform';
    const text = `${message.subject}\n${message.body}`;
    const affects = watched.filter(run => text.includes(run.run_id) || text.includes(run.unit)).map(run => run.run_id);
    const asks = kind === 'answer-request' ? 'asks for an answer' : kind === 'adopt-decision' ? 'announces a decision' : 'informs';
    return { message_id: message.id, kind, what: `${sentence(message.from)} ${asks}: ${sentence(message.subject)}.`, affects, reply_needed: kind === 'answer-request' || message.urgent };
  });
  const byId = new Map(messages.map(message => [message.id, message]));
  const ordered = [...obligations].sort((a, b) => URGENT_FIRST(a, b, byId));
  const digest = ordered.length ? ordered.slice(0, 3).map(item => item.what).join(' ') : 'No messages need this machine.';
  return { obligations, digest };
}

export type PlanArgs = { diagnoses: RunDiagnosis[], triage: InboxTriage, resources: Resources, next_steps: Untrusted<string>[], allowlist: ActionSummary[], history: CycleMemory[] };
export type PlanAux = { defs: ActionDef[], settings: Settings, machine: string, repo: string, runs: RunEvidence[] };

const ORDER: Health[] = ['finished-failed', 'blocked', 'gate-failed', 'stalled', 'unknown', 'progressing', 'finished-ok'];
const healthy = (health: Health) => health === 'progressing' || health === 'finished-ok';

export function idleResource(resources: Resources, idle: Settings['idle']): string {
  const headroom = resources.headroom_gb !== null && resources.headroom_gb >= idle.headroom_gb;
  const gpu = resources.gpu_utilization === null || resources.gpu_utilization <= idle.gpu_utilization;
  const teacher = resources.teacher_load === null || resources.teacher_load <= idle.teacher_load;
  if (!(headroom && gpu && teacher)) return '';
  return resources.gpu_utilization === null ? 'memory' : 'gpu';
}

const sameProposal = (a: Proposal, b: Proposal) => a.action === b.action && a.target === b.target && JSON.stringify(a.params) === JSON.stringify(b.params);

function defaultParam(name: string, kind: string, run: RunEvidence | undefined, minutesBelow: number): string {
  if (kind === 'minutes') return String(Math.max(1, Math.min(15, minutesBelow - 1)));
  if (kind === 'number') return String(run?.ledger_claim?.budget_gb ?? 4);
  if (kind === 'watched-unit') return run?.entry.unit ?? '';
  if (kind === 'run-id') return run?.entry.run_id ?? '';
  return name;
}

export function crispPlan(args: PlanArgs, aux: PlanAux): Plan {
  const proposals: Proposal[] = [];
  const last = args.history[0]?.proposals ?? [];
  const byRun = new Map(aux.runs.map(run => [run.entry.run_id, run]));
  const proposable = aux.defs.filter(def => def.proposable);
  const make = (def: ActionDef, run: RunEvidence | undefined, message: string | undefined, why: string, cites: string[]): Proposal => ({
    action: def.id,
    target: def.target === 'run' ? run!.entry.run_id : def.target === 'unit' ? run!.entry.unit : def.target === 'message' ? message! : aux.machine,
    params: Object.fromEntries(Object.entries(def.params).map(([name, kind]) => [name, defaultParam(name, kind, run, aux.settings.timer_interval_minutes)])),
    why, cites, expected_effect: def.what });
  const ordered = [...args.diagnoses].sort((a, b) => ORDER.indexOf(a.health) - ORDER.indexOf(b.health));
  for (const diagnosis of ordered.filter(item => !healthy(item.health))) {
    const run = byRun.get(diagnosis.run_id);
    if (!run) continue;
    // Reversible evidence-gathering first, then actions that change the run; an unchanged proposal of the last cycle gives way to the next one.
    const matching = proposable.filter(def => def.applies_to.includes(diagnosis.cause) && (def.target === 'run' || def.target === 'unit' || def.target === 'machine'))
      .sort((a, b) => Number(b.reversible) - Number(a.reversible));
    for (const def of matching) {
      const proposal = make(def, run, undefined, `${diagnosis.run_id} is ${diagnosis.health} with cause ${diagnosis.cause}.`, [diagnosis.run_id]);
      if (!last.some(item => sameProposal(item, proposal)) && !proposals.some(item => sameProposal(item, proposal))) proposals.push(proposal);
    }
  }
  const reply = proposable.find(def => def.id === 'reply-note');
  if (reply) for (const obligation of args.triage.obligations.filter(item => item.reply_needed)) {
    const proposal = make(reply, undefined, obligation.message_id, `Message ${obligation.message_id} needs an answer.`, [obligation.message_id]);
    proposal.params = { text: `Seen by the ${aux.machine} heartbeat; the machine session answers.` };
    if (!last.some(item => sameProposal(item, proposal))) proposals.push(proposal);
  }
  const idle = idleResource(args.resources, aux.settings.idle);
  const queue = proposable.find(def => def.id === 'queue-next');
  if (idle && queue) {
    for (const text of args.next_steps) {
      const header = /^Next steps of run (\S+):\n([^]*)$/.exec(text);
      const step = header?.[2]?.split('\n').map(line => line.replace(/^\s*(?:[-*]|\d+[.)])\s*/, '').trim()).find(line => line !== '');
      if (!header || !step || !byRun.has(header[1]!)) continue;
      const proposal = make(queue, byRun.get(header[1]!), undefined, `The ${idle} is idle and ${header[1]} has a written next step.`, [header[1]!]);
      proposal.params = { step: clip(step, 500) };
      if (!last.some(item => sameProposal(item, proposal))) proposals.push(proposal);
      break;
    }
  }
  const failed = args.diagnoses.filter(item => !healthy(item.health)).length;
  const summary = `${args.diagnoses.length} run(s) watched, ${failed} not healthy. ${proposals.length} action(s) proposed.${idle ? ` The ${idle} is idle.` : ''}`;
  return { proposals, idle_resources: idle, summary };
}

