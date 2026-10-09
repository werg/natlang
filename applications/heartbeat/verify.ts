// Exact verifiers of the stage outputs. Each returns the problems of an answer as sentences that say what a correct
// answer has; the cycle hands them back to the function (feedback) and the answer is repaired, or replaced by the crisp side.
import { gateFailures } from './collect/gate.js';
import { idleResource } from './ladder.js';
import { proposalProblems, type ActionDef, type Known, type Settings } from './actions.js';
import type { CoordMessage, InboxTriage, Obligation, Plan, Resources, RunDiagnosis, RunEvidence, WatchedRun } from './types.js';

/** How many sentences a text has: runs of text ended by a full stop, question or exclamation mark, or the end. */
export function sentences(text: string): number {
  return text.split(/(?<=[.!?])\s+(?=[A-Z0-9"'`(])/).filter(part => part.trim() !== '').length;
}

export function diagnosisProblems(diagnosis: RunDiagnosis, run: RunEvidence): string[] {
  const problems: string[] = [];
  if (diagnosis.run_id !== run.entry.run_id) problems.push(`run_id is "${run.entry.run_id}"`);
  if (!diagnosis.evidence.length && run.readings.length) problems.push('evidence has one to three entries');
  if (diagnosis.evidence.length > 3) problems.push('evidence has at most three entries');
  for (const item of diagnosis.evidence) {
    const reading = run.readings.find(candidate => candidate.id === item.reading_id);
    if (!reading) problems.push(`evidence reading_id "${item.reading_id}" is the id of one of the readings (${run.readings.map(candidate => candidate.id).join(', ')})`);
    else if (!item.quote.trim() || !reading.text.includes(item.quote)) problems.push(`the quote ${JSON.stringify(item.quote.slice(0, 80))} is copied letter for letter from the text of reading ${reading.id}`);
  }
  if (gateFailures(run.gates).length && diagnosis.health !== 'gate-failed') problems.push('health is "gate-failed" when a gate field reports failure');
  if (diagnosis.health === 'gate-failed' && !gateFailures(run.gates).length) problems.push('health is "gate-failed" only when a gate field reports failure');
  if (sentences(diagnosis.summary) !== 1) problems.push('summary is one sentence');
  else if (!diagnosis.summary.includes(run.entry.run_id)) problems.push(`summary names the run "${run.entry.run_id}"`);
  return problems;
}

export function triageProblems(triage: InboxTriage, messages: CoordMessage[], watched: WatchedRun[]): string[] {
  const problems: string[] = [];
  const ids = new Set(messages.map(message => message.id));
  const seen = new Set<string>();
  for (const obligation of triage.obligations) {
    if (!ids.has(obligation.message_id)) problems.push(`message_id "${obligation.message_id}" is the id of one of the messages`);
    if (seen.has(obligation.message_id)) problems.push(`message ${obligation.message_id} has one obligation`);
    seen.add(obligation.message_id);
    if (sentences(obligation.what) !== 1) problems.push(`the "what" of ${obligation.message_id} is one sentence`);
    const message = messages.find(candidate => candidate.id === obligation.message_id);
    if (message && (obligation.kind === 'answer-request' || message.urgent) && !obligation.reply_needed) problems.push(`reply_needed is true for ${obligation.message_id}`);
    if (message && message.kind === 'request' && obligation.kind !== 'answer-request') problems.push(`the request ${obligation.message_id} has kind "answer-request"`);
  }
  for (const id of ids) if (!seen.has(id)) problems.push(`message ${id} has an obligation`);
  void watched;
  if (sentences(triage.digest) > 3) problems.push('digest has at most three sentences');
  return problems;
}

export type PlanCheck = { defs: ActionDef[], known: Known, diagnoses: RunDiagnosis[], triage: InboxTriage, resources: Resources, settings: Settings };

/** Problems of each proposal (null: valid) and of the plan as a whole. */
export function planProblems(plan: Plan, check: PlanCheck): { proposals: (string[] | null)[], plan: string[] } {
  const cites = new Set([...check.diagnoses.map(item => item.run_id), ...check.triage.obligations.map((item: Obligation) => item.message_id)]);
  const proposals = plan.proposals.map(proposal => {
    const problems = proposalProblems(proposal, check.defs, check.known);
    for (const cite of proposal.cites) if (!cites.has(cite)) problems.push(`cites lists ids of the diagnoses and obligations (${[...cites].join(', ') || 'none'}), not "${cite}"`);
    if (!proposal.cites.length) problems.push('cites names the run or message the proposal answers');
    if (sentences(proposal.why) !== 1) problems.push('why is one sentence');
    return problems.length ? problems.map(problem => `${proposal.action} on ${proposal.target}: ${problem}`) : null;
  });
  const whole: string[] = [];
  const idle = idleResource(check.resources, check.settings.idle);
  if (idle && !plan.idle_resources) whole.push(`idle_resources names the idle resource "${idle}" (headroom, gpu and teacher load are at their idle levels)`);
  if (!idle && plan.idle_resources) whole.push('idle_resources is empty because no resource is idle');
  if (sentences(plan.summary) > 3) whole.push('summary has at most three sentences');
  return { proposals, plan: whole };
}
