import type { SearchCandidate } from '../types.js';
/** The comparison strategy always mutates the incumbent; evaluator/proposer/budgets are shared. */
export function reflectionParent(population: readonly SearchCandidate[], incumbent: string): SearchCandidate {
  const parent = population.find(candidate => candidate.id === incumbent);
  if (!parent) throw new Error('incumbent missing from population'); return parent;
}
