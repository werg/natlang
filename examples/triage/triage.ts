/**
 * Support-ticket triage: `escalate` wraps the local scored decision so unsure tickets go to a stronger model,
 * `is_urgent.nl` flags urgent tickets, ordinary TypeScript does the exact bookkeeping, and `semanticRepair`
 * shortens the refined inline `nl` summary until it fits the word budget.
 */
import { nl } from '@natlang/node';
import { escalate, semanticRepair } from '@natlang/node/std';
import classify from './classify.nl';
import isUrgent from './is_urgent.nl';
import type { Label, Report, Summary } from './types.js';

export type { Label, Report };

const words = (text: string) => text.trim() ? text.trim().split(/\s+/).length : 0;
const MAX_SUMMARY_WORDS = 60;

export async function triage(tickets: Untrusted<string>[], rubric: string): Promise<Report> {
  // Only the tickets the local model is unsure of go to the stronger model.
  const classifyTicket = escalate(classify, { to: 'teacher', floor: 0.8 });
  const labels = (await Promise.all(tickets.map(ticket => classifyTicket(ticket, rubric)))).map(result => result.value);
  const real = tickets.filter((_, i) => labels[i] !== 'spam');
  const flags = await Promise.all(real.map(ticket => isUrgent(ticket)));
  const urgent = real.filter((_, i) => flags[i]);
  const by_label: Record<string, number> = {};
  for (const label of labels) by_label[label] = (by_label[label] ?? 0) + 1;
  return { urgent: urgent.length, by_label, summary: urgent.length ? await summarize(urgent) : 'Nothing urgent today.' };
}

/** One short paragraph about the urgent tickets, most severe first. */
async function summarize(tickets: Untrusted<string>[]): Promise<Summary> {
  const draft: Summary = await nl`Write one paragraph saying what is going wrong across tickets, most severe first.`(tickets);
  const shorten = nl<(text: Summary, problems: string[]) => Summary>`Rewrite text to be noticeably shorter so none of problems applies. Keep every fact; drop filler and repetition.`;
  const repaired = await semanticRepair(draft, {
    check: text => words(text) <= MAX_SUMMARY_WORDS ? [] : [`Summary has ${words(text)} words; shorten to at most ${MAX_SUMMARY_WORDS} words.`],
    fix: shorten,
    attempts: 3,
  });
  return repaired.value;
}
