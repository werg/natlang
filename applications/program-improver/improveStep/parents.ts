import type {PopulationMember} from '../types';
/** GEPA per-case frontier archive, with seeded sampling weighted by validation wins. */
export function frontierParent(population: PopulationMember[], seed: number): string {
  const cases = [...new Set(population.flatMap(member => (member.scores ?? []).map(score => score.caseId)))].sort();
  const winners: string[] = [];
  for (const caseId of cases) {
    const maximum = Math.max(...population.map(member => member.scores?.find(score => score.caseId === caseId)?.quality ?? -1));
    for (const member of population) if (member.scores?.find(score => score.caseId === caseId)?.quality === maximum) winners.push(member.source);
  }
  const choices = winners.length ? winners.sort() : population.map(member => member.source).sort();
  let random = seed >>> 0; random ^= random << 13; random ^= random >>> 17; random ^= random << 5;
  return choices[(random >>> 0) % choices.length];
}
