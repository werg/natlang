import { join } from 'node:path';
import { RecipeTerminal } from '../semantic_terminal.mjs';
import { natlangWorkspaceRecipes } from '../terminal_recipes.mjs';

const empty = () => ({ revision: 0, active_request: '', active_job: '', status: 'idle', messages: [], history: [] });

/** Trusted native edge for the packaged semantic terminal. Its reducer and view remain natlang programs. */
export async function createTarget(context) {
  if (!context.modelTurn) throw new Error('semantic terminal needs a configured model profile');
  const { NativeNatlangHost, TerminalEventQueue, TerminalNatlangApplication,
    TerminalSessionStore, runTerminalShell } = context.runtime;
  const library = natlangWorkspaceRecipes(context.workspace);
  const terminal = new RecipeTerminal(library.recipes());
  const completions = new TerminalEventQueue();
  const unsubscribe = terminal.subscribe(event => completions.push(event));
  const host = new NativeNatlangHost({ host: { terminal, drainEvents: () => terminal.drainEvents() }, mode: 'retained' });
  const store = new TerminalSessionStore(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(empty());
  if (checkpoint.state.status === 'running' || checkpoint.state.status === 'cancel-requested')
    completions.push({ kind: 'recover', id: `recover-${checkpoint.revision}`,
      request_id: checkpoint.state.active_request, job_id: checkpoint.state.active_job,
      text: '', status: 'unknown', detail: 'host process restarted' });
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: join(context.package.root, ...context.target.reducer.split('/')),
      view: join(context.package.root, ...context.target.view.split('/')) },
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, seedRoot: 17, traceDirectory: context.traceDirectory,
    modelTurn: context.modelTurn, onCommit: commit => store.commit(commit, app.seenEventIds) });
  return {
    async run() {
      await runTerminalShell(app, { input: context.io.input, output: context.io.output, events: completions,
        color: context.io.color,
        event: (text, id) => ({ kind: 'request', id, request_id: '', job_id: '', text, status: '', detail: '' }),
        cancelEvent: id => ({ kind: 'cancel', id, request_id: app.state.active_request,
          job_id: app.state.active_job, text: '', status: '', detail: '' }) });
      return 0;
    },
    async close() { unsubscribe(); completions.close(); terminal.close(); await app.close(); host.close(); },
  };
}
