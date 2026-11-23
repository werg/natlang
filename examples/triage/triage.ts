/**
 * Support-ticket triage: a scored decision labels each ticket (the unsure ones go to a stronger model), another judges
 * urgency, ordinary TypeScript does the exact bookkeeping, and an inline `nl` summary, whose property is its type, is
 * shortened with `iterateOn` until it fits.
 */
import { decide, IterationLimitError, iterateOn, nl } from '@natlang/node';
import classify from './classify.nl';
import isUrgent from './is_urgent.nl';
import type { Label, Report, Summary } from './types.js';

export type { Label, Report };

const words = (text: string) => text.trim() ? text.trim().split(/\s+/).length : 0;
const MAX_SUMMARY_WORDS = 60;

export async function triage(tickets: string[], rubric: string): Promise<Report> {
  const decisions = await Promise.all(tickets.map(ticket => decide(classify, ticket, rubric)));
  // Only the tickets the local model is unsure of go to the stronger model.
  const labels = await Promise.all(decisions.map(({ value, confidence }, i) =>
    confidence >= 0.8 ? value : classify.on({ model: 'teacher' })(tickets[i]!, rubric)));
  const real = tickets.filter((_, i) => labels[i] !== 'spam');
  const flags = await Promise.all(real.map(ticket => isUrgent(ticket)));
  const urgent = real.filter((_, i) => flags[i]);
  const by_label: Record<string, number> = {};
  for (const label of labels) by_label[label] = (by_label[label] ?? 0) + 1;
  return { urgent: urgent.length, by_label, summary: urgent.length ? await summarize(urgent) : 'Nothing urgent today.' };
}

/** One short paragraph about the urgent tickets, most severe first. */
async function summarize(tickets: string[]): Promise<Summary> {
  const draft: Summary = await nl`Write one paragraph saying what is going wrong across tickets, most severe first.`(tickets);
  const shorten = async (text: Summary): Promise<Summary> =>
    nl<Summary>`Rewrite text to be noticeably shorter. Keep every fact; drop filler and repetition.`(text);
  try {
    return await iterateOn(shorten, draft).withLimit({ maxSteps: 3 }).until(text => words(text) <= MAX_SUMMARY_WORDS);
  } catch (error) {
    // After three rewrites, keep the shortest honest summary rather than failing the report.
    if (error instanceof IterationLimitError) return error.lastState as Summary;
    throw error;
  }
}
