// The applier: the only part that acts. It carries out a proposal only when the allowlist row says the action may run by
// itself at the current stage (reversible, not process control, auto "always", or "promoted" at stage auto), with argv
// fixed by the row and parameters validated against it. Everything else stays a proposal to the owning session. An
// applied action writes a receipt: the state before, the argv, the exit code and the state after.
import { autoApplies, commandFor, proposalProblems, type ActionDef, type Known, type Settings, type Stage } from './actions.js';
import type { Host } from './host.js';
import { sha256 } from './collect/reading.js';
import type { Plan, Proposal } from './types.js';

export type Receipt = {
  argv: string[] | null,
  started_at: string,
  exit_code: number | null,
  before: { sha256: string, text: string } | null,
  after: { sha256: string, text: string } | null,
  output: string,
};
export type Disposition = {
  proposal: Proposal,
  /** applied: carried out; failed: attempted and the command failed; proposed: left to the owner of the action; rejected: invalid against the allowlist. */
  status: 'applied' | 'failed' | 'proposed' | 'rejected',
  owner: string,
  reversible: boolean | null,
  /** Why it was not applied, or what was done. */
  reason: string,
  command: string[] | null,
  receipt: Receipt | null,
};

const bounded = (text: string, length = 1500) => text.length > length ? `${text.slice(0, length)} … (${text.length} chars)` : text;
const state = (text: string) => ({ sha256: sha256(text), text: bounded(text) });

async function probe(host: Host, def: ActionDef, proposal: Proposal, known: Known, cycleArgv: string[]) {
  if (!def.probe) return null;
  const argv = commandFor({ ...def, argv: def.probe }, proposal, known, cycleArgv);
  const result = await host.exec(argv!, { timeoutMs: 30_000, cwd: known.repo, env: { COORD_MACHINE: known.machine } });
  return state(result.ok ? result.stdout : `probe failed: ${result.stderr}`);
}

/** Carry out one proposal, or leave it. `plan` supplies the summary for actions that take it on standard input. */
export async function applyOne(host: Host, proposal: Proposal, defs: ActionDef[], known: Known, settings: Settings, stage: Stage, plan: Plan): Promise<Disposition> {
  const def = defs.find(item => item.id === proposal.action);
  const invalid = proposalProblems(proposal, defs, known);
  if (!def || invalid.length) return { proposal, status: 'rejected', owner: def?.owner ?? 'unknown', reversible: def?.reversible ?? null, reason: invalid.join('; '), command: null, receipt: null };
  const command = commandFor(def, proposal, known, settings.cycle_argv);
  if (!autoApplies(def, stage)) {
    const why = def.process_control ? 'process control is decided by the owning session' : !def.reversible ? 'it cannot be undone, so the owner decides'
      : def.auto === 'never' ? 'the allowlist does not let it run by itself' : `it runs by itself only at stage auto (the stage is ${stage})`;
    return { proposal, status: 'proposed', owner: def.owner, reversible: def.reversible, reason: why, command, receipt: null };
  }
  if (!command) return { proposal, status: 'proposed', owner: def.owner, reversible: def.reversible, reason: 'the action has no command to run', command, receipt: null };
  const before = await probe(host, def, proposal, known, settings.cycle_argv);
  const started = host.now().toISOString();
  const result = await host.exec(command, { timeoutMs: 120_000, cwd: known.repo, env: { COORD_MACHINE: known.machine }, input: def.stdin === 'plan-summary' ? plan.summary : undefined });
  const after = await probe(host, def, proposal, known, settings.cycle_argv);
  const receipt: Receipt = { argv: command, started_at: started, exit_code: result.code, before, after, output: bounded(result.stdout || result.stderr) };
  return { proposal, status: result.ok ? 'applied' : 'failed', owner: def.owner, reversible: def.reversible, reason: result.ok ? 'applied automatically' : bounded(result.stderr || 'the command failed', 300), command, receipt };
}

/** Apply a plan's proposals in order. Proposals do not depend on each other, so a failure does not stop the rest. */
export async function applyPlan(host: Host, plan: Plan, defs: ActionDef[], known: Known, settings: Settings, stage: Stage): Promise<Disposition[]> {
  const done: Disposition[] = [];
  for (const proposal of plan.proposals) done.push(await applyOne(host, proposal, defs, known, settings, stage, plan));
  return done;
}
