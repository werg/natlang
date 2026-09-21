import { createInterface } from 'node:readline';
import { join } from 'node:path';
import { LogWorkspace } from '../log_investigator.mjs';

export function createTarget(context) {
  if (!context.modelTurn) throw new Error('log console needs a configured model profile');
  const { NativeNatlangHost, TerminalNatlangApplication, TerminalSessionStore, renderTerminalView } = context.runtime;
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
  return { async run() {
      context.io.output.write(renderTerminalView((await app.start()).view,
        { color: context.io.color, width: context.io.output.columns }));
      await app.consume(lines(), transition => context.io.output.write(renderTerminalView(transition.view,
        { color: context.io.color, width: context.io.output.columns })));
      return 0;
    }, async close() { await app.close(); host.close(); } };
}
