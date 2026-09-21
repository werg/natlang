/*---
engine: typescript-host
args:
  state: IncidentState
returns: TerminalView
---*/
const state = args.state;
return { title: 'Natlang Log Investigator', subtitle: `${state.observed} records observed through cursor ${state.cursor}`,
  blocks: [
    { kind: 'status', text: `Status: ${state.status}`, tone:
      state.status === 'alerted' ? 'bad' : state.status === 'delivery-unknown' || state.status === 'gap' ? 'warn' : 'muted' },
    ...(state.alerts.length ? [{ kind: 'table', columns: ['Status', 'Incident', 'Detail'],
      rows: state.alerts.slice(-10).map(row => [row.status, row.key, row.detail]) }] : []),
    ...(state.unknowns.length ? [{ kind: 'list', items: state.unknowns.slice(-10) }] : []),
  ], help: ['Consumes typed JSON log events', 'State commits after each event'] };
