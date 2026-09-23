/** Log Console: a guided or streaming (stdin JSONL) semantic log anomaly tracker. */
import { readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { join, resolve } from 'node:path';
import { EventLoop, TerminalSessionStore, openFolder, renderTerminalView, runTerminalShell,
  type TargetContext, type TerminalView } from '@natlang/node';
import { LogWorkspace, emptyIncidentState, step, type IncidentState, type LogEvent } from './index.js';

export function view(state: IncidentState): TerminalView {
  return { title: 'Natlang Log Investigator', subtitle: `${state.observed} records observed through cursor ${state.cursor}`,
    blocks: [
      { kind: 'status', text: `Status: ${state.status}`, tone:
        state.status === 'alerted' ? 'bad' : state.status === 'delivery-unknown' || state.status === 'gap' ? 'warn' : 'muted' },
      ...(state.observed === 0 ? [{ kind: 'text' as const, tone: 'muted' as const, text:
        'Ready for logs. Use /demo for an immediate incident walkthrough, /load FILE for JSONL, paste a JSON event, or type a plain informational record.' }] : []),
      ...(state.alerts.length ? [{ kind: 'table' as const, columns: ['Status', 'Incident', 'Detail'],
        rows: state.alerts.slice(-10).map(row => [row.status, row.key, row.detail]) }] : []),
      ...(state.unknowns.length ? [{ kind: 'list' as const, items: state.unknowns.slice(-10) }] : []),
    ], prompt: 'logs> ', help: ['/help commands', '/demo sample incident', '/load FILE ingest JSONL', '/quit exit'] };
}

type Event = LogEvent & { [key: string]: unknown };

export async function main(context: TargetContext): Promise<number> {
  const logs = new LogWorkspace();
  const store = new TerminalSessionStore<IncidentState, Event>(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(emptyIncidentState());
  const files = () => openFolder(context.workspace).root();
  const loop: EventLoop<IncidentState, TerminalView, Event> = new EventLoop({
    initialState: checkpoint.state, initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    reduce: (state, event) => step(logs, state, event, files()), view,
    step: fn => context.runtime.run(fn), onCommit: commit => store.commit(commit, loop.seenEventIds) });
  let nextCursor = checkpoint.state.cursor + 1;
  const event = (value: string, id: string): Event => {
    if (value.startsWith('{')) {
      const parsed = JSON.parse(value) as Event;
      const cursor = parsed.cursor ?? nextCursor++;
      nextCursor = Math.max(nextCursor, Number(cursor) + 1);
      return { ...parsed, id: parsed.id ?? id, cursor };
    }
    return { kind: 'log', id, cursor: nextCursor++, occurred_at: Date.now(), arrived_at: Date.now(),
      service: 'manual', code: 'NOTE', level: 'info', message: value };
  };
  const demo = () => {
    const now = Date.now(), batch = `demo-${now}`;
    return [0, 1, 2].map((index): Event => ({ kind: 'log', id: `${batch}-${index + 1}`, cursor: nextCursor++,
      occurred_at: now + index * 1000, arrived_at: now + index * 1000, service: 'checkout', code: 'PAYMENT_TIMEOUT',
      level: 'error', message: `Payment request timed out for independent request ${index + 1}` }));
  };
  const output = context.io.output as NodeJS.WriteStream;
  const show = (value: TerminalView) => output.write(renderTerminalView(value, { color: context.io.color, width: output.columns }));
  try {
    if ((context.io.input as NodeJS.ReadStream).isTTY) {
      await runTerminalShell(loop, { input: context.io.input as never, output: context.io.output as never, color: context.io.color, event,
        commands: {
          demo: { description: 'ingest a three-event incident demonstration', run: demo },
          load: { description: 'FILE ingest a JSONL log file', run: value => {
            if (!value) throw new Error('provide a JSONL path');
            return readFileSync(resolve(context.workspace, value), 'utf8').split(/\r?\n/).filter(line => line.trim())
              .map((line, index) => event(line, `file-${Date.now()}-${index}`));
          } },
          example: { description: 'show accepted interactive input', run: () =>
            'Enter plain text for a manual informational record, paste one JSON log event, use /demo, or use /load FILE.' },
        } });
      return 0;
    }
    async function* lines(): AsyncIterable<Event> {
      let index = 0;
      for await (const line of createInterface({ input: context.io.input as never }))
        if (line.trim()) yield event(line.trim(), `stdin-${index++}`);
    }
    show((await loop.start()).view);
    await loop.consume(lines(), transition => { show(transition.view); });
    return 0;
  } finally { await loop.close(); }
}
