/**
 * Slot sizing for the managed local model server (plans/BATCHED_EXECUTION.md §3.2).
 *
 * llama.cpp divides `-c` among `--parallel` slots, so N slots of C tokens each need `-c N*C`. The slot count is
 * therefore the number of C-token KV caches that fit in the memory we may spend beyond the model's weights:
 *
 *   default:  kvBudget = min(availableMemory / 2, 4 GiB);  slots = clamp(floor(kvBudget / (contextTokens * kvBytesPerToken)), 1, maxSlots)
 *   explicit (local.memoryBudgetMiB / NATLANG_MODEL_MEMORY_BUDGET_MIB, unbounded; covers weights and KV):
 *             slots = clamp(floor((budget - modelBytes) / (contextTokens * kvBytesPerToken)), 1, maxSlots)
 *
 * The default cap exists because on unified-memory machines (DGX) admitted jobs claim memory after the server starts.
 *
 * Defaults: memoryFraction 0.5, kvBytesPerToken 65536 (64 KiB, a conservative figure for sub-2B models; set
 * `local.kvBytesPerToken` or NATLANG_MODEL_KV_BYTES_PER_TOKEN for a model whose cache is larger), maxSlots 8. An
 * explicit `local.parallel` always wins. On unified-memory machines (DGX) available memory is the GPU's, so the
 * budget must be agreed with the GPU ledger: set `local.memoryBudgetMiB` to the ledger's grant.
 */
import { freemem } from 'node:os';
import type { ModelProfile } from './config.js';

export type SlotPlan = { budgetSource: 'configured-parallel' | 'memoryBudgetMiB' | 'NATLANG_MODEL_MEMORY_BUDGET_MIB' | 'default';
  slots: number; contextPerSlot: number; totalContext: number; source: 'configured' | 'memory';
  budgetBytes: number | null; modelBytes: number; kvBytesPerToken: number; formula: string };

export const DEFAULT_KV_BYTES_PER_TOKEN = 65536;
export const DEFAULT_MEMORY_FRACTION = 0.5;
export const DEFAULT_MAX_SLOTS = 8;
/** The default KV budget never exceeds this, so a server started from momentary free memory cannot crowd out later jobs. */
export const DEFAULT_KV_BUDGET_CAP_BYTES = 4 * 2 ** 30;

export function planServerSlots(input: { local?: ModelProfile['local']; modelBytes: number; defaultContextTokens: number;
  availableMemoryBytes?: number; environment?: NodeJS.ProcessEnv }): SlotPlan {
  const { local, modelBytes } = input, environment = input.environment ?? process.env;
  const contextPerSlot = local?.contextTokens ?? input.defaultContextTokens;
  const kvBytesPerToken = local?.kvBytesPerToken ?? (Number(environment.NATLANG_MODEL_KV_BYTES_PER_TOKEN) || DEFAULT_KV_BYTES_PER_TOKEN);
  const envBudget = Number(environment.NATLANG_MODEL_MEMORY_BUDGET_MIB) || 0;
  const budgetMiB = local?.memoryBudgetMiB ?? (envBudget > 0 ? envBudget : undefined);
  const available = input.availableMemoryBytes ?? freemem();
  const explicit = budgetMiB !== undefined;
  const budgetBytes = explicit ? budgetMiB * 2 ** 20 : Math.min(Math.floor(available * DEFAULT_MEMORY_FRACTION), DEFAULT_KV_BUDGET_CAP_BYTES);
  if (local?.parallel !== undefined)
    return { budgetSource: 'configured-parallel', slots: local.parallel, contextPerSlot, totalContext: contextPerSlot * local.parallel, source: 'configured',
      budgetBytes: null, modelBytes, kvBytesPerToken, formula: 'local.parallel' };
  // An explicit budget covers weights and KV (and is unbounded); the default budget is for the KV caches alone.
  const fit = Math.floor((budgetBytes - (explicit ? modelBytes : 0)) / (contextPerSlot * kvBytesPerToken));
  const slots = Math.max(1, Math.min(DEFAULT_MAX_SLOTS, Number.isFinite(fit) ? fit : 1));
  return { budgetSource: !explicit ? 'default' : local?.memoryBudgetMiB !== undefined ? 'memoryBudgetMiB' : 'NATLANG_MODEL_MEMORY_BUDGET_MIB',
    slots, contextPerSlot, totalContext: contextPerSlot * slots, source: 'memory', budgetBytes, modelBytes, kvBytesPerToken,
    formula: `clamp(floor((${budgetBytes} - ${explicit ? modelBytes : 0}) / (${contextPerSlot} * ${kvBytesPerToken})), 1, ${DEFAULT_MAX_SLOTS}) = ${slots}` };
}
