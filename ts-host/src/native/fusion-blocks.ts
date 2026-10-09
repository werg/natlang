/**
 * Blocks written by fused hand-offs (runtime/fusion.ts). A fused edge keeps a producer's value as a Neuralese block while
 * the author-visible declared type stays `T`; value checks therefore accept such a block where `T` is declared. The set
 * is per task, so a block of the same content elsewhere is not accepted by accident.
 */
import { isNeuraleseRef } from './neuralese.js';

const FUSED = new WeakMap<object, Set<string>>();
let currentTask: () => object | undefined = () => undefined;

/** The runtime installs how to find the task a value check runs in (kept out of this module's imports: it loads early). */
export function setCurrentTaskResolver(resolver: () => object | undefined): void { currentTask = resolver; }

/** Record that `task` holds the block `id` as a fused hand-off. */
export function markFusedBlock(task: object, id: string): void {
  let blocks = FUSED.get(task);
  if (!blocks) FUSED.set(task, blocks = new Set());
  blocks.add(id);
}

/** Whether `raw` is a Neuralese reference to a block the current task holds as a fused hand-off. */
export function isFusedRef(raw: unknown): boolean {
  if (!isNeuraleseRef(raw)) return false;
  const task = currentTask();
  return !!task && !!FUSED.get(task)?.has(raw.$neuralese.id);
}
