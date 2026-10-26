import type { ProtocolMandate, ResourceAssessment, AssignmentSearch, InstrumentWindow, ResearchOperationsSnapshot } from '../types';

/** Exhaustively enumerate a finite assignment space; never relax protocol predicates. */
export default function findOpenSlots(mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot): AssignmentSearch {
  const windows = snapshot.instrumentWindows ?? [];
  const bookings = snapshot.bookings ?? [];
  const assessments = new Map(resources.map(item => [item.experimentId, item]));
  const ordered = [...mandates].sort((a, b) => {
    const ca = assessments.get(a.experimentId)?.eligibleInstrumentIds.length ?? 0;
    const cb = assessments.get(b.experimentId)?.eligibleInstrumentIds.length ?? 0;
    return ca - cb || a.experimentId.localeCompare(b.experimentId);
  });
  const options = new Map<string, InstrumentWindow[]>();
  for (const mandate of ordered) {
    const eligible = new Set(assessments.get(mandate.experimentId)?.eligibleInstrumentIds ?? []);
    options.set(mandate.experimentId, windows.filter(w => eligible.has(w.instrumentId) &&
      w.capabilities.includes(mandate.requiredCapability) && w.qualificationRefs.length > 0 &&
      Date.parse(w.calibrationValidUntil) >= Date.parse(mandate.sampleWindowStart) &&
      Date.parse(w.startsAt) < Date.parse(mandate.sampleWindowEnd) && Date.parse(w.endsAt) > Date.parse(mandate.sampleWindowStart)));
  }
  const picked: Array<{ experimentId: string; instrument: InstrumentWindow; starts: number; ends: number }> = [];
  let best: typeof picked = [];
  let exploredBranches = 0;
  function search(index: number): void {
    if (picked.length > best.length) best = [...picked];
    if (index === ordered.length || picked.length + ordered.length - index < best.length) return;
    const mandate = ordered[index]!;
    const duration = mandate.durationHours * 3_600_000;
    for (const instrument of options.get(mandate.experimentId) ?? []) {
      const earliest = Math.max(Date.parse(mandate.sampleWindowStart), Date.parse(instrument.startsAt));
      const latest = Math.min(Date.parse(mandate.sampleWindowEnd), Date.parse(instrument.endsAt)) - duration;
      const starts = [...new Set([earliest, ...picked.filter(item => item.instrument.instrumentId === instrument.instrumentId ||
        item.instrument.operatorId === instrument.operatorId).map(item => item.ends),
        ...bookings.filter(booking => booking.status === 'booked' &&
          (booking.instrumentId === instrument.instrumentId || booking.operatorId === instrument.operatorId)).map(item => Date.parse(item.endsAt))])].sort((a, b) => a - b);
      for (const start of starts) {
        const end = start + duration;
        if (!Number.isFinite(start) || start < earliest || start > latest) continue;
        exploredBranches++;
        const overlaps = (from: number, to: number, otherFrom: number, otherTo: number) => to > otherFrom && otherTo > from;
        const collision = picked.some(item => overlaps(start, end, item.starts, item.ends) &&
          (item.instrument.instrumentId === instrument.instrumentId || item.instrument.operatorId === instrument.operatorId)) ||
          bookings.some(booking => booking.status === 'booked' && overlaps(start, end, Date.parse(booking.startsAt), Date.parse(booking.endsAt)) &&
            (booking.instrumentId === instrument.instrumentId || booking.operatorId === instrument.operatorId));
        if (collision) continue;
        const assignment = { experimentId: mandate.experimentId, instrument, starts: start, ends: end };
        picked.push(assignment); search(index + 1); picked.pop();
      }
    }
    exploredBranches++;
    search(index + 1);
  }
  search(0);
  const result = ordered.flatMap(m => {
    const item = best.find(candidate => candidate.experimentId === m.experimentId);
    if (!item) return [];
    return [{ experimentId: m.experimentId, instrumentId: item.instrument.instrumentId,
      operatorId: item.instrument.operatorId, startsAt: new Date(item.starts).toISOString(), endsAt: new Date(item.ends).toISOString(),
      protocolId: m.protocolId, measurementBehavior: item.instrument.measurementBehavior, deviationStatus: 'none' as const,
      sourceRefs: m.sourceRefs }];
  });
  const assigned = new Set(best.map(item => item.experimentId));
  const unscheduledExperimentIds = ordered.map(m => m.experimentId).filter(id => !assigned.has(id));
  return { assignments: result, unscheduledExperimentIds,
    conflictSets: [], searchComplete: true, exploredBranches };
}
