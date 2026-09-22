/*---
engine: typescript-host
args:
  state: IncidentState
  files?: Dict<File>
returns: TerminalView
---*/
const state = args.state;
return { title: 'Natlang Log Investigator', subtitle: `${state.observed} records observed through cursor ${state.cursor}`,
  blocks: [
    { kind: 'status', text: `Status: ${state.status}`, tone:
      state.status === 'alerted' ? 'bad' : state.status === 'delivery-unknown' || state.status === 'gap' ? 'warn' : 'muted' },
    ...(state.observed === 0 ? [{ kind: 'text', text: 'Ready for logs. Use /demo for an immediate incident walkthrough, /load FILE for JSONL, paste a JSON event, or type a plain informational record.', tone: 'muted' }] : []),
    ...(state.alerts.length ? [{ kind: 'table', columns: ['Status', 'Incident', 'Detail'],
      rows: state.alerts.slice(-10).map(row => [row.status, row.key, row.detail]) }] : []),
    ...(state.unknowns.length ? [{ kind: 'list', items: state.unknowns.slice(-10) }] : []),
  ], prompt: 'logs> ', help: ['/help commands', '/demo sample incident', '/load FILE ingest JSONL', '/quit exit'] };
