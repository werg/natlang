/**
 * The ready tasks: needed, not finished, and every dependency finished. A pluggable hot path of every round. The
 * setting `build.implementation('ready')` selects this file's filter (crisp) or `ready/rule.nl` (natural-language).
 */
import { pluggable } from '@natlang/node';
import { build } from 'natlang:services';
import rule from './ready/rule.nl';
import type { Graph } from '../../types.js';

/** The ids of the ready tasks, sorted. */
export function crisp(graph: Graph, needed: string[], finished: string[]): string[] {
  const done = new Set(finished), want = new Set(needed);
  return graph.nodes.filter(node => want.has(node.id) && !done.has(node.id) && node.deps.every(dep => done.has(dep)))
    .map(node => node.id).sort();
}

export default async function ready(graph: Graph, needed: string[], finished: string[]): Promise<string[]> {
  return pluggable({ crisp: () => crisp(graph, needed, finished), nl: () => rule(graph, needed, finished) },
    await build.implementation('ready'), { name: 'build.ready' })();
}
