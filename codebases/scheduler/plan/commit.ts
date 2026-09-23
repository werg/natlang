export default function commit(chosen: Candidate, revision: number): ScheduleResult {
return host.scheduler.commit(chosen, revision);
}
