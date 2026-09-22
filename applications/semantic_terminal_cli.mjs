import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost, TerminalEventQueue, TerminalNatlangApplication, TerminalSessionStore,
  NodeFileTree, runTerminalShell } from '../ts-host/dist/index.js';
import { RecipeTerminal } from './semantic_terminal.mjs';
import { natlangWorkspaceRecipes } from './terminal_recipes.mjs';
import { cliFlag, modelTurnFromCli } from './natlang_cli.mjs';

const here = fileURLToPath(new URL('.', import.meta.url));
const source = name => fileURLToPath(new URL(`../codebases/semantic_terminal/${name}`, import.meta.url));
export const emptyTerminalSession = () => ({ revision: 0, active_request: '', active_job: '',
  status: 'idle', messages: [], history: [] });

export async function runSemanticTerminal({ root = resolve(here, '..'), sessionPath,
  traceDirectory, modelTurn, recipes, input, output, seedRoot = 17 } = {}) {
  if (!modelTurn) throw new Error('semantic terminal requires a modelTurn driver');
  const library = recipes ?? natlangWorkspaceRecipes(root);
  const terminal = new RecipeTerminal(library.recipes());
  const completions = new TerminalEventQueue();
  const unsubscribe = terminal.subscribe(event => completions.push(event));
  const native = { terminal, drainEvents: () => terminal.drainEvents() };
  const host = new NatlangHost({ host: native, mode: 'retained' });
  const store = sessionPath ? new TerminalSessionStore(sessionPath) : null;
  const checkpoint = store?.load(emptyTerminalSession()) ?? { revision: 0,
    state: emptyTerminalSession(), seen_event_ids: [] };
  if (checkpoint.state.status === 'running' || checkpoint.state.status === 'cancel-requested')
    completions.push({ kind: 'recover', id: `recover-${checkpoint.revision}`,
      request_id: checkpoint.state.active_request, job_id: checkpoint.state.active_job,
      text: '', status: 'unknown', detail: 'host process restarted' });
  let app;
  app = new TerminalNatlangApplication({ runner: host,
    source: { reducer: source('reduce.nl'), view: source('view.ts') },
    inputs: { files: new NodeFileTree(root) },
    initialState: checkpoint.state, initialRevision: checkpoint.revision,
    seenEventIds: checkpoint.seen_event_ids, seedRoot, traceDirectory, modelTurn,
    onCommit: commit => store?.commit(commit, app.seenEventIds) });
  try {
    await runTerminalShell(app, { input, output, events: completions,
      event: (text, id) => ({ kind: 'request', id, request_id: '', job_id: '', text,
        status: '', detail: '' }),
      cancelEvent: id => ({ kind: 'cancel', id, request_id: app.state.active_request,
        job_id: app.state.active_job, text: '', status: '', detail: '' }),
      commands: {
        recipes: { description: 'list exact operations natlang can choose', run: () => library.recipes()
          .map(recipe => `${recipe.id}  ${recipe.description}`).join('\n') },
        example: { description: 'show example requests', run: () =>
          'Inspect the repository status and summarize what changed.\nRun the TypeScript test suite and explain any failure.' },
      } });
  } finally { unsubscribe(); completions.close(); terminal.close(); host.close(); }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  const root = resolve(cliFlag(args, '--root', '.'));
  const sessionPath = resolve(cliFlag(args, '--session', '.natlang/terminal-session.json'));
  const traceDirectory = resolve(cliFlag(args, '--traces', '.natlang/traces'));
  const modelTurn = modelTurnFromCli(args);
  await runSemanticTerminal({ root, sessionPath, traceDirectory, modelTurn });
}
