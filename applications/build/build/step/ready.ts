/**
 * The ready tasks: needed, not finished, and every dependency finished. A pluggable hot path of every round. The
 * setting `build.implementation('ready')` selects this file's filter (crisp) or `ready/rule.nl` (natural-language).
 */
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
  if (await build.implementation('ready') === 'natural-language') return rule(graph, needed, finished);
  return crisp(graph, needed, finished);
}
