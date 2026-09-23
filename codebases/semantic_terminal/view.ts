import type { Event, File, Recipe, Job, Outcome, Session, ViewBlock, TerminalView } from "./types.js";

export default function view(state: Session): TerminalView {
const latest = state.messages.slice(-8);
const blocks = [
  { kind: 'status', text: `Status: ${state.status}`, tone:
      state.status === 'ok' ? 'good' : state.status === 'failed' || state.status === 'unknown' ? 'bad' :
      state.status === 'running' || state.status === 'cancel-requested' ? 'warn' : 'muted' },
  ...(latest.length ? [{ kind: 'list', items: latest }] : [
    { kind: 'text', text: 'Describe a workspace task. Natlang will choose from the exact available recipes.', tone: 'muted' },
    { kind: 'list', items: ['Try: Inspect the repository status and summarize what changed.', 'Use /recipes to see operations or /help for controls.'] },
  ]),
];
if (state.history.length) blocks.push({ kind: 'table', columns: ['Request', 'Job', 'Status', 'Detail'],
  rows: state.history.slice(-8).map(row => [row.request_id, row.job_id, row.status, row.detail]) });
return { title: 'Natlang Terminal', subtitle: `Session revision ${state.revision}`, blocks,
  prompt: state.status === 'running' ? 'job running> ' : 'natlang> ',
  busy: state.status === 'running' || state.status === 'cancel-requested',
  help: ['/help commands', '/recipes capabilities', '/cancel request job cancellation', '/quit exit'] };
}
