import { join } from 'node:path';

/** Common lifecycle for packaged natlang terminal reducers. Domain objects stay application owned. */
export function terminalExecutable(context, { hostObject, initialState, event, events, cancelEvent, commands, close, inputs, reducerInputs, viewInputs }) {
  if (!context.modelTurn) throw new Error(`${context.package.name} needs a configured model profile`);
  const { NativeNatlangHost, TerminalNatlangApplication, TerminalSessionStore, runTerminalShell } = context.runtime;
  const host = new NativeNatlangHost({ host: hostObject, mode: 'retained' });
  const store = new TerminalSessionStore(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(initialState());
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: join(context.package.root, ...context.target.reducer.split('/')),
      view: join(context.package.root, ...context.target.view.split('/')) },
    inputs, reducerInputs, viewInputs,
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, seedRoot: 17, traceDirectory: context.traceDirectory,
    modelTurn: context.modelTurn, onCommit: commit => store.commit(commit, app.seenEventIds) });
  return { async run() { await runTerminalShell(app, { input: context.io.input, output: context.io.output,
      color: context.io.color, event, events, cancelEvent,
      commands: typeof commands === 'function' ? commands(app) : commands }); return 0; },
    async close() { await app.close(); host.close(); await close?.(); } };
}

export function flag(args, name) {
  const index = args.indexOf(name);
  return index < 0 ? undefined : args[index + 1];
}
