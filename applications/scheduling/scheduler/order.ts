import type { Domain, Ordering } from '../types.js';

/**
 * The tasks in an order where each one follows the tasks it depends on: depth first from each task in the order given,
 * a task's dependencies (in their listed order) before the task. When a dependency names a missing task or the
 * dependencies form a cycle, `order` is empty and `problem` says which tasks are involved. This is the one ordering of the
 * scheduling app: the verifier (`Problem.order`) and the stages use it, so they enumerate in the same order.
 */
export function topologicalOrder(tasks: { task: string, after: string[] }[]): Ordering {
  const after = new Map(tasks.map(item => [item.task, item.after]));
  for (const item of tasks) {
    const unknown = item.after.find(id => !after.has(id));
    if (unknown) return { order: [], problem: `${item.task} depends on ${unknown}, which is not a task of the day` };
  }
  const done = new Set<string>(), sorted: string[] = [], path: string[] = [];
  let cycle: string[] | null = null;
  const visit = (id: string) => {
    if (cycle || done.has(id)) return;
    const at = path.indexOf(id);
    if (at >= 0) { cycle = path.slice(at); return; }
    path.push(id);
    for (const dependency of after.get(id)!) visit(dependency);
    path.pop();
    if (cycle) return;
    done.add(id); sorted.push(id);
  };
  for (const item of tasks) visit(item.task);
  if (cycle) return { order: [], problem: `${(cycle as string[]).join(', ')} depend on each other in a cycle` };
  return { order: sorted, problem: '' };
}

/** The tasks of domains in dependency order (see topologicalOrder). */
export default function order(domains: Domain[]): Ordering {
  return topologicalOrder(domains);
}
