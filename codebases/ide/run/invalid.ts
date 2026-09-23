export default function invalid(revision: string, checked: CheckReport): RunReport {
return { status: 'invalid-source', run_id: '', revision: revision,
  source_revision: '',
  value_text: '', trace_events: 0, detail: checked.detail };
}
