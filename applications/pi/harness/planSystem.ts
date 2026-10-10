/**
 * The pi.system entries one request needs (§7.4). Pluggable hot path, selected by the host setting planning: `crisp`
 * (default) pi-durable's planSystemEntries (through durable.planSystem); `nl` `planSystem/rules.nl`, the same rules in
 * natural language; `shadow` both, compared.
 */
import { pluggable } from 'natlang:runtime';
import { durable } from 'natlang:services';
import rules from './planSystem/rules.nl';
import type { AgentTool, ContextEdit, ContextView, SystemMessage } from '../types.js';

export default async function planSystem(view: ContextView, desired: { key: string; text: string }[], tools: AgentTool[], now: number):
  Promise<{ message: SystemMessage; edits?: ContextEdit[] }[]> {
  return pluggable({ crisp: () => durable.planSystem(view, desired, tools, now), nl: () => rules(view, desired, tools, now) },
    await durable.implementation('planning'), { default: 'crisp', name: 'pi.planSystem' })();
}
