/** Semantic Terminal: describe a workspace task; natlang picks an exact recipe and explains the outcome. */
import { join } from 'node:path';
import { EventLoop, EventQueue, TerminalSessionStore, openFolder, runTerminalShell, type TargetContext, type TerminalView } from '@natlang/node';
import { RecipeTerminal, emptyTerminalSession, natlangWorkspaceRecipes, step, type Session, type TerminalEvent } from './index.js';

export function view(state: Session): TerminalView {
  const latest = state.messages.slice(-8);
  const blocks: TerminalView['blocks'] = [
    { kind: 'status', text: `Status: ${state.status}`, tone: state.status === 'ok' ? 'good' :
      state.status === 'failed' || state.status === 'unknown' ? 'bad' :
      state.status === 'running' || state.status === 'cancel-requested' ? 'warn' : 'muted' },
    ...(latest.length ? [{ kind: 'list' as const, items: latest }] : [
      { kind: 'text' as const, text: 'Describe a workspace task. Natlang will choose from the exact available recipes.', tone: 'muted' as const },
      { kind: 'list' as const, items: ['Try: Inspect the repository status and summarize what changed.', 'Use /recipes to see operations or /help for controls.'] }]),
  ];
  if (state.history.length) blocks.push({ kind: 'table', columns: ['Request', 'Job', 'Status', 'Detail'],
    rows: state.history.slice(-8).map(row => [row.request_id, row.job_id, row.status, row.detail]) });
  return { title: 'Natlang Terminal', subtitle: `Session revision ${state.revision}`, blocks,
    prompt: state.status === 'running' ? 'job running> ' : 'natlang> ', busy: state.status === 'running' || state.status === 'cancel-requested',
    help: ['/help commands', '/recipes capabilities', '/cancel request job cancellation', '/quit exit'] };
}

export async function main(context: TargetContext): Promise<number> {
  const library = natlangWorkspaceRecipes(context.workspace);
  const terminal = new RecipeTerminal(library.recipes());
  const completions = new EventQueue<TerminalEvent>();
  const unsubscribe = terminal.subscribe(event => completions.push(event));
  const store = new TerminalSessionStore<Session, TerminalEvent>(join(context.stateDirectory, 'session.json'));
  const checkpoint = store.load(emptyTerminalSession());
  if (checkpoint.state.status === 'running' || checkpoint.state.status === 'cancel-requested')
    completions.push({ kind: 'recover', id: `recover-${checkpoint.revision}`, request_id: checkpoint.state.active_request,
      job_id: checkpoint.state.active_job, text: '', status: 'unknown', detail: 'host process restarted' });
  const loop: EventLoop<Session, TerminalView, TerminalEvent> = new EventLoop({
    initialState: checkpoint.state, initialRevision: checkpoint.revision, seenEventIds: checkpoint.seen_event_ids,
    reduce: (state, event) => step(terminal, state, event, openFolder(context.workspace).root()), view,
    step: fn => context.runtime.run(fn), onCommit: commit => store.commit(commit, loop.seenEventIds) });
  try {
    await runTerminalShell(loop, { input: context.io.input as never, output: context.io.output as never, color: context.io.color,
      events: completions,
      event: (text, id) => ({ kind: 'request', id, request_id: '', job_id: '', text, status: '', detail: '' }),
      cancelEvent: id => ({ kind: 'cancel', id, request_id: loop.state.active_request, job_id: loop.state.active_job,
        text: '', status: '', detail: '' }),
      commands: {
        recipes: { description: 'list exact operations natlang can choose', run: () => library.recipes()
          .map(recipe => `${recipe.id}  ${recipe.description}`).join('\n') },
        example: { description: 'show example requests', run: () => ['Inspect the repository status and summarize what changed.',
          'Run the TypeScript test suite and explain any failure.'].join('\n') },
      } });
  } finally { unsubscribe(); completions.close(); terminal.close(); await loop.close(); }
  return 0;
}
