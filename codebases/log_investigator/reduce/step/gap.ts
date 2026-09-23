export default function gap(acc: IncidentState, item: LogEvent): IncidentState {
if (item.cursor <= acc.cursor) return acc;
return { ...acc, cursor: item.cursor, status: 'gap',
  unknowns: [...acc.unknowns, `Source gap at cursor ${item.cursor}: ${item.message}`] };
}
