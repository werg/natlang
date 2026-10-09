// The heartbeat record: one JSON line per cycle in runs/heartbeat/<date>.jsonl (the whole trajectory: evidence in,
// diagnoses, plan, what was applied) and runs/heartbeat/latest.md, rendered by crisp code from the same record.
import type { Disposition } from './apply.js';
import type { Stage, Mode } from './actions.js';
import type { AgentAction, CycleSummary } from './compare.js';
import type { Reading } from './collect/reading.js';
import type { Host } from './host.js';
import type { CycleMemory, InboxTriage, Plan, RunDiagnosis } from './types.js';

export type Via = 'nl' | 'crisp' | 'crisp-fallback' | 'crisp-degraded' | 'shadow';
export type StageNote = { via: Via, attempts: number, problems: string[], error?: string, agree?: boolean, nl_summary?: string };

export type CycleRecord = {
  schema: 'natlang.heartbeat-cycle/1',
  cycle: string,
  at: string,
  machine: string,
  stage: Stage,
  modes: Record<'diagnoseRun' | 'triageInbox' | 'planNext', Mode>,
  degraded: { executor_unreachable: boolean, natural_language_unavailable: boolean },
  /** Readings keep the mark of outside text: their text is data, never instructions. */
  readings: (Omit<Reading, 'text'> & { text: string, untrusted: true })[],
  runs: { run_id: string, unit: string, diagnosis: RunDiagnosis, note: StageNote }[],
  triage: InboxTriage & { note: StageNote },
  plan: Plan & { note: StageNote, rejected: { proposal: unknown, problems: string[] }[] },
  dispositions: Disposition[],
  unwatched: string[],
  warnings: string[],
  /** What the agents did since the previous cycle (derived from observable effects). */
  agent_actions: AgentAction[],
};

export const dayOf = (at: string) => at.slice(0, 10);

export async function appendRecord(host: Host, repo: string, dir: string, record: CycleRecord): Promise<string> {
  const path = `${repo}/${dir}/${dayOf(record.at)}.jsonl`;
  await host.appendText(path, `${JSON.stringify(record)}\n`);
  return path;
}

/** The records of the last days, oldest first. Damaged lines are skipped. */
export async function readRecords(host: Host, repo: string, dir: string, machine: string, days: string[]): Promise<CycleRecord[]> {
  const found: CycleRecord[] = [];
  for (const day of days) {
    const file = await host.readText(`${repo}/${dir}/${day}.jsonl`);
    for (const line of file?.text.split('\n').filter(Boolean) ?? []) try {
      const record = JSON.parse(line) as CycleRecord;
      if (record.schema === 'natlang.heartbeat-cycle/1' && record.machine === machine) found.push(record);
    } catch { /* skip */ }
  }
  return found.sort((a, b) => a.at.localeCompare(b.at));
}

export function memoryOf(records: CycleRecord[]): CycleMemory[] {
  return records.map((record, index) => ({ cycle: record.cycle, proposals: record.plan.proposals,
    agent_actions: (records[index + 1]?.agent_actions ?? []).map(action => ({ action: action.action, target: action.target })) })).reverse();
}

export const summaryOf = (record: CycleRecord): CycleSummary => ({ cycle: record.cycle, at: record.at,
  runs: record.runs.map(run => ({ run_id: run.run_id, unit: run.unit, health: run.diagnosis.health, cause: run.diagnosis.cause })), proposals: record.plan.proposals });

const cell = (text: string) => text.replace(/\|/g, '/').replace(/\n/g, ' ');

export function renderReport(record: CycleRecord): string {
  const lines: string[] = [
    `# Heartbeat ${record.cycle}`, '',
    `Machine ${record.machine}, ${record.at}, stage **${record.stage}**` +
      (record.stage === 'auto' ? ' (reversible allowlisted actions apply by themselves)' : record.stage === 'advisory' ? ' (advisory: only the record is written)' : ' (shadow: only the record is written)') + '.',
  ];
  if (record.degraded.natural_language_unavailable) lines.push('', '**Natural language was unavailable this cycle; the crisp status ladder produced these findings.**');
  else if (record.degraded.executor_unreachable) lines.push('', '**The executor did not answer; the crisp status ladder produced these findings.**');
  for (const warning of record.warnings) lines.push('', `Warning: ${warning}`);
  lines.push('', '## Summary', '', record.plan.summary);
  lines.push('', '## Runs', '');
  if (!record.runs.length) lines.push('No runs are declared in watch.json for this machine.');
  else {
    lines.push('| run | unit | health | cause | confidence | by |', '| --- | --- | --- | --- | --- | --- |');
    for (const run of record.runs) lines.push(`| ${cell(run.run_id)} | ${cell(run.unit)} | ${run.diagnosis.health} | ${run.diagnosis.cause} | ${run.diagnosis.confidence} | ${run.note.via} |`);
    for (const run of record.runs.filter(item => item.diagnosis.health !== 'progressing' && item.diagnosis.health !== 'finished-ok'))
      lines.push('', `- **${run.run_id}**: ${run.diagnosis.summary}` + run.diagnosis.evidence.map(item => `\n  - ${item.reading_id}: \`${cell(item.quote.slice(0, 160))}\``).join(''));
  }
  if (record.unwatched.length) lines.push('', `Unwatched (running or claimed without a watch entry): ${record.unwatched.join(', ')}.`);
  lines.push('', '## Inbox', '', record.triage.digest);
  for (const obligation of record.triage.obligations.filter(item => item.kind !== 'inform'))
    lines.push(`- ${obligation.message_id} (${obligation.kind}${obligation.reply_needed ? ', reply needed' : ''}): ${obligation.what}`);
  lines.push('', '## Proposals', '');
  const proposals = record.dispositions.filter(item => item.proposal.action !== 'record-heartbeat');
  if (!proposals.length) lines.push('None.');
  for (const item of proposals) {
    lines.push(`- **${item.proposal.action}** on ${item.proposal.target} - ${item.status} (owner ${item.owner}): ${item.proposal.why}`);
    if (item.status !== 'applied') lines.push(`  - ${item.reason}`);
    if (item.command) lines.push(`  - \`${item.command.join(' ')}\``);
  }
  if (record.plan.idle_resources) lines.push('', `Idle: ${record.plan.idle_resources}.`);
  if (record.plan.rejected.length) lines.push('', `${record.plan.rejected.length} proposal(s) of the planner failed the allowlist checks and were dropped.`);
  const failing = record.readings.filter(reading => !reading.ok);
  if (failing.length) lines.push('', '## Readings that failed', '', ...failing.map(reading => `- ${reading.id}: ${cell(reading.text.slice(0, 200))}`));
  return `${lines.join('\n')}\n`;
}

export async function writeLatest(host: Host, repo: string, dir: string, record: CycleRecord): Promise<string> {
  const path = `${repo}/${dir}/latest.md`;
  await host.writeText(path, renderReport(record));
  return path;
}
