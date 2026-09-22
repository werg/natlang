import { createInterface } from 'node:readline';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { LogWorkspace } from '../log_investigator.mjs';

export function createTarget(context) {
  if (!context.modelTurn) throw new Error('log console needs a configured model profile');
  const { NativeNatlangHost, TerminalNatlangApplication, TerminalSessionStore,
    renderTerminalView, runTerminalShell } = context.runtime;
  const logs = new LogWorkspace(), host = new NativeNatlangHost({ host: { logs,
    drainEvents: () => logs.drainEvents() }, mode: 'retained' });
  const empty = () => ({ cursor: -1, observed: 0, alerts: [], unknowns: [], status: 'idle' });
  const store = new TerminalSessionStore(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(empty());
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: join(context.package.root, ...context.target.reducer.split('/')),
      view: join(context.package.root, ...context.target.view.split('/')) }, initialState: checkpoint.state,
    initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    modelTurn: context.modelTurn, seedRoot: 17, traceDirectory: context.traceDirectory,
    onCommit: commit => store.commit(commit, app.seenEventIds) });
  async function* lines() {
    for await (const line of createInterface({ input: context.io.input })) if (line.trim()) yield JSON.parse(line);
  }
  let nextCursor = checkpoint.state.cursor + 1;
  const event = (value, id) => {
    if (value.startsWith('{')) {
      const parsed = JSON.parse(value);
      const cursor = parsed.cursor ?? nextCursor++;
      nextCursor = Math.max(nextCursor, Number(cursor) + 1);
      return { ...parsed, id: parsed.id ?? id, cursor };
    }
    return { kind: 'log', id, cursor: nextCursor++, occurred_at: Date.now(), arrived_at: Date.now(),
      service: 'manual', code: 'NOTE', level: 'info', message: value };
  };
  const demo = () => {
    const now = Date.now(), batch = `demo-${now}`;
    return [0, 1, 2].map(index => ({ kind: 'log', id: `${batch}-${index + 1}`, cursor: nextCursor++,
      occurred_at: now + index * 1000, arrived_at: now + index * 1000,
      service: 'checkout', code: 'PAYMENT_TIMEOUT', level: 'error',
      message: `Payment request timed out for independent request ${index + 1}` }));
  };
  return { async run() {
      if (context.io.input.isTTY) {
        await runTerminalShell(app, { input: context.io.input, output: context.io.output,
          color: context.io.color, event,
          commands: {
            demo: { description: 'ingest a three-event incident demonstration', run: demo },
            load: { description: 'FILE ingest a JSONL log file', run: value => {
              if (!value) throw new Error('provide a JSONL path');
              return readFileSync(resolve(context.workspace, value), 'utf8').split(/\r?\n/)
                .filter(line => line.trim()).map((line, index) => event(line, `file-${Date.now()}-${index}`));
            } },
            example: { description: 'show accepted interactive input', run: () =>
              'Enter plain text for a manual informational record, paste one JSON log event, use /demo, or use /load FILE.' },
          } });
        return 0;
      }
      context.io.output.write(renderTerminalView((await app.start()).view,
        { color: context.io.color, width: context.io.output.columns }));
      await app.consume(lines(), transition => context.io.output.write(renderTerminalView(transition.view,
        { color: context.io.color, width: context.io.output.columns })));
      return 0;
    }, async close() { await app.close(); host.close(); } };
}
