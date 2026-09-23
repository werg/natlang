export default function infeasible(snapshot: ScheduleSnapshot, alternatives: Alternatives): ScheduleResult {
return { status: 'infeasible', revision: snapshot.revision,
  plan: snapshot.plan, detail: alternatives.detail };
}
