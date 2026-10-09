import type { Domain, Ordering } from '../types.js';

/**
 * The tasks in an order where each one follows the tasks it depends on. When a dependency names a missing task or the
 * dependencies form a cycle, `order` is empty and `problem` says which tasks are involved.
 */
export default function order(domains: Domain[]): Ordering {
  const known = new Set(domains.map(domain => domain.task));
  for (const domain of domains) {
    const unknown = domain.after.find(id => !known.has(id));
    if (unknown) return { order: [], problem: `${domain.task} depends on ${unknown}, which is not a task of the day` };
  }
  const placed = new Set<string>(), sorted: string[] = [];
  for (let round = 0; round < domains.length; round++) {
    for (const domain of domains) {
      if (!placed.has(domain.task) && domain.after.every(id => placed.has(id))) { placed.add(domain.task); sorted.push(domain.task); }
    }
  }
  if (sorted.length < domains.length)
    return { order: [], problem: `${domains.filter(domain => !placed.has(domain.task)).map(domain => domain.task).join(', ')} depend on each other in a cycle` };
  return { order: sorted, problem: '' };
}
