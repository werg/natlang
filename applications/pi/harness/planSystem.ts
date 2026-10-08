/**
 * The pi.system entries one request needs (§7.4). Pluggable hot path ("planning"): pi-durable's crisp
 * planSystemEntries (through durable.planSystem) by default; with planning "natural-language", `planSystem/rules.nl`, the same rules in natural
 * language.
 */
import { durable } from 'natlang:services';
import rules from './planSystem/rules.nl';
import type { AgentTool, ContextEdit, ContextView, SystemMessage } from '../types.js';

export default async function planSystem(view: ContextView, desired: { key: string; text: string }[], tools: AgentTool[], now: number):
  Promise<{ message: SystemMessage; edits?: ContextEdit[] }[]> {
  if (durable.implementation('planning') === 'natural-language') return rules(view, desired, tools, now);
  return durable.planSystem(view, desired, tools, now);
}
