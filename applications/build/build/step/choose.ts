/**
 * Which ready task runs next: the build's scheduling policy, a pluggable hot path. The setting
 * `build.implementation('choose')` selects the smallest id (crisp) or `choose/pick.nl` (natural-language), which weighs
 * the tasks' descriptions and reads a named input from files.
 */
import { build } from 'natlang:services';
import pick from './choose/pick.nl';
import type { FolderHandle } from '@natlang/node';
import type { Task } from '../../types.js';

/** The smallest id among the ready tasks. */
export function crisp(ready: Task[]): string {
  return ready.map(task => task.id).sort()[0] ?? '';
}

export default async function choose(ready: Task[], goal: string, files?: FolderHandle): Promise<string> {
  if (await build.implementation('choose') === 'natural-language') return pick(ready, goal, files);
  return crisp(ready);
}
