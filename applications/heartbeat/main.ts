/**
 * `natlang run applications/heartbeat -- [cycle|collect|report|compare|record] [options]`
 *
 * The hourly check-in as a program (plans/HEARTBEAT_PROGRAM.md). `cycle` (the default) collects evidence, diagnoses each
 * watched run, triages the coordination inbox, plans the next actions and records the report; the allowlist and the stage in
 * actions.json decide what, if anything, is applied. `collect` prints the evidence only. `report` prints runs/heartbeat/latest.md.
 * `compare` compares the recorded proposals with what the agents did. `record --action ID --target T` notes an agent action that
 * left no visible effect.
 */
import type { TargetContext } from '@natlang/node';
import { loadActions, loadWatch, type Stage } from './actions.js';
import { compare, deriveAgentActions, recordAction } from './compare.js';
import { collect } from './collect/index.js';
import { nodeHost, type Host } from './host.js';
import { runCycle, waitForIdle } from './heartbeat.js';
import { readRecords, summaryOf } from './record.js';

const STAGES: Stage[] = ['shadow', 'advisory', 'auto'];

function parse(args: string[]) {
  const value = (name: string) => { const at = args.indexOf(name); return at >= 0 ? args[at + 1] : undefined; };
  const command = args[0] && !args[0].startsWith('--') ? args[0] : 'cycle';
  const stage = value('--stage');
  if (stage !== undefined && !STAGES.includes(stage as Stage)) throw new Error(`--stage is one of ${STAGES.join(', ')}`);
  return { command, stage: stage as Stage | undefined, record: !args.includes('--no-record'), repo: value('--repo'), config: value('--config'),
    action: value('--action'), target: value('--target'), days: Number(value('--days') ?? 7), noWait: args.includes('--no-wait') };
}

const daysBack = (host: Host, count: number) => Array.from({ length: count }, (_, index) => new Date(host.now().getTime() - index * 86_400_000).toISOString().slice(0, 10)).reverse();
const ledgerPath = () => process.env.NATLANG_MEMORY_LEDGER ?? `${process.env.HOME ?? ''}/.local/state/natlang/memory-ledger.json`;

export async function main(context: TargetContext): Promise<number> {
  const options = parse(context.args);
  const host = nodeHost();
  const out = (text: string) => context.io.output.write(`${text}\n`);
  const configDir = options.config ?? context.package?.root ?? process.cwd();
  const { value: actions } = await loadActions(host, `${configDir}/actions.json`);
  const settings = actions.settings;
  const repo = options.repo ?? settings.repo ?? process.env.HEARTBEAT_REPO ?? process.cwd();
  const machine = host.machine();

  if (options.command === 'collect') {
    const snapshot = await collect(host, settings, repo, await loadWatch(host, `${configDir}/watch.json`, machine));
    out(JSON.stringify(snapshot, null, 2));
    return 0;
  }
  if (options.command === 'report') {
    const latest = await host.readText(`${repo}/${settings.record_dir}/latest.md`);
    out(latest?.text ?? 'No heartbeat report yet.');
    return latest ? 0 : 1;
  }
  if (options.command === 'record') {
    if (!options.action || !options.target) { context.io.error.write('record needs --action ID and --target T\n'); return 2; }
    if (!actions.actions.some(def => def.id === options.action)) { context.io.error.write(`--action is one of ${actions.actions.map(def => def.id).join(', ')}\n`); return 2; }
    out(JSON.stringify(await recordAction(host, repo, settings.record_dir, options.action, options.target)));
    return 0;
  }
  if (options.command === 'compare') {
    const records = await readRecords(host, repo, settings.record_dir, machine, daysBack(host, options.days));
    if (!records.length) { out('No heartbeat cycles recorded yet.'); return 0; }
    const watch = await loadWatch(host, `${configDir}/watch.json`, machine);
    const agentActions = await deriveAgentActions(host, { repo, machine, since: records[0]!.at, until: host.now().toISOString(), watch,
      ignoreUnits: settings.ignore_units.map(pattern => new RegExp(pattern)), ledgerPath: ledgerPath(), recordDir: settings.record_dir });
    out(JSON.stringify(compare(records.map(summaryOf), agentActions, watch), null, 2));
    return 0;
  }
  if (options.command !== 'cycle') { context.io.error.write(`unknown command ${options.command}; use cycle, collect, report, compare or record\n`); return 2; }

  const log = (line: string) => out(`${new Date().toISOString()} ${line}`);
  const endpoint = context.modelEndpoint?.endpoint ?? settings.executor_endpoint;
  if (!options.noWait) await waitForIdle(host, endpoint, settings, log, ms => new Promise(done => setTimeout(done, ms)));
  const record = await runCycle({ host, run: fn => context.runtime.run(fn), configDir, repo, stage: options.stage, executorEndpoint: endpoint, record: options.record, log });
  out(`${record.cycle}: ${record.runs.length} run(s), ${record.plan.proposals.length} proposal(s), ${record.dispositions.filter(item => item.status === 'applied' && item.proposal.action !== 'record-heartbeat').length} applied`);
  return 0;
}
