/**
 * Crisp check of a hindsight hint (DECOMPOSITION.md, part "hint check"): the exact verifier of what `writeHint`
 * returns. A hint is procedure and evidence for a call that computes its own result, within a declared token budget.
 * It fails when it is empty, echoes the writer's own signature, is over budget, or copies a value the verified
 * execution produced: its returned value, or a number it computed (one shown by a non-reading step that the prompt
 * and the reading steps never showed; 0 and 1 are too common to tell). The failure is one sentence the writer reads as its tool error.
 */
import type { TrajectoryCall, TrajectoryStep } from './types.js';

/** The declared hint budget, in tokens (estimated as characters / CHARS_PER_TOKEN). */
export const HINT_BUDGET_TOKENS = 64;
export const CHARS_PER_TOKEN = 4;
/** Steps that only read inputs: what they show is evidence, not a produced value. */
const READING_TOOLS = new Set(['read_file', 'list_files', 'read_code', 'read_inputs', 'search_files', 'grep_files']);

const NUMBER = /^[-+]?\d+(?:\.\d+)?$/;
const text = (value: unknown): string => typeof value === 'string' ? value : JSON.stringify(value) ?? '';
/** A step's output without the runtime's cut-off notes (their counts are not values of the run). */
const output = (step: TrajectoryStep): string => text(step.output).replace(/<<cut off:[^>]*>>/g, ' ');

/** The numeric tokens of a text, as numbers ("R-21", "C-101" and list markers are not numbers). */
export function numbersIn(value: string): number[] {
  return value.replace(/^\s*\d+[.)]\s/gm, ' ').split(/[\s,;:()[\]{}"'`|=<>/*]+/)
    .map(token => token.replace(/[.!?]+$/, '')).filter(token => NUMBER.test(token)).map(Number);
}

/** The scalar leaves of a returned value (numbers, numeric strings as numbers, other strings of at least 3 characters). */
function leaves(value: unknown, out: (number | string)[] = []): (number | string)[] {
  if (typeof value === 'number' && Number.isFinite(value)) out.push(value);
  else if (typeof value === 'string' && NUMBER.test(value.trim())) out.push(Number(value.trim()));
  else if (typeof value === 'string' && value.trim().length >= 3) out.push(value.trim());
  else if (Array.isArray(value)) for (const item of value) leaves(item, out);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) leaves(item, out);
  return out;
}

/** What the execution returned: `result.value`, else its last `return_result` step's value. */
function returned(execution: TrajectoryCall): unknown {
  if (execution.result && 'value' in execution.result) return execution.result.value;
  const last = [...execution.steps].reverse().find((step: TrajectoryStep) => step.tool === 'return_result');
  return last?.arguments?.value;
}

/** The values execution produced that a hint must not copy: returned leaves and computed numbers. */
export function producedValues(execution: TrajectoryCall): { numbers: Set<number>; strings: Set<string> } {
  const shown = new Set(numbersIn(text(execution.prompt)));
  for (const step of execution.steps) {
    if (step.tool !== 'return_result') for (const number of numbersIn(text(step.arguments))) shown.add(number);
    if (READING_TOOLS.has(step.tool)) for (const number of numbersIn(output(step))) shown.add(number);
  }
  const numbers = new Set<number>(), strings = new Set<string>();
  for (const step of execution.steps)
    if (!READING_TOOLS.has(step.tool) && step.tool !== 'return_result')
      for (const number of numbersIn(output(step))) if (!shown.has(number) && number !== 0 && number !== 1) numbers.add(number);
  const prompt = text(execution.prompt);
  for (const leaf of leaves(returned(execution))) {
    if (typeof leaf === 'number') { if (!numbersIn(prompt).includes(leaf)) numbers.add(leaf); }
    else if (!prompt.includes(leaf)) strings.add(leaf);
  }
  return { numbers, strings };
}

/** Why `hint` is rejected, as one sentence, or undefined when it passes. */
export function checkHint(hint: unknown, execution: TrajectoryCall, budgetTokens: number = HINT_BUDGET_TOKENS): string | undefined {
  if (typeof hint !== 'string' || !hint.trim()) return 'Return the hint as nonempty text.';
  if (/\bwriteHint\s*\(/.test(hint)) return 'Return only the hint text, without this function\'s name or signature.';
  const tokens = Math.ceil(hint.trim().length / CHARS_PER_TOKEN);
  if (tokens > budgetTokens)
    return `The hint is about ${tokens} tokens, over its ${budgetTokens}-token limit (about ${budgetTokens * CHARS_PER_TOKEN} ` +
      'characters): keep only the steps and evidence the call needs.';
  const { numbers, strings } = producedValues(execution);
  const copied = [...numbersIn(hint).filter(number => numbers.has(number)).map(String),
    ...[...strings].filter(value => hint.includes(value))];
  if (copied.length)
    return `The hint gives ${[...new Set(copied)].slice(0, 3).join(', ')}, which execution produced; the call computes ` +
      'its own results, so name how to find them and the evidence to read instead.';
  return undefined;
}

/**
 * Crisp check of any written guidance (writeGuidance's or writeHint's result) against the values the run withholds
 * (rl/privileged_context.py `answer_leaves`: the world's expected result, reference return values and child values;
 * the writer never sees them). Guidance says where a call's result comes from; the call computes it. A number counts
 * as a whole number token, a string as a case-insensitive whole word or phrase. One sentence, or undefined.
 */
export function checkGuidance(guidance: unknown, withheld: readonly string[]): string | undefined {
  if (typeof guidance !== 'string' || !withheld.length) return undefined;
  const numbers = new Set(numbersIn(guidance));
  const lower = guidance.toLowerCase();
  const escape = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const given = withheld.filter(value => NUMBER.test(value.trim()) ? numbers.has(Number(value)) :
    new RegExp(`(?<![\\p{L}\\p{N}_])${escape(value.toLowerCase())}(?![\\p{L}\\p{N}_])`, 'u').test(lower));
  if (!given.length) return undefined;
  return `The guidance gives ${given.length} value${given.length === 1 ? '' : 's'} of the call's result; say where the ` +
    'result comes from (which input, file, function or step) and let the call compute it, without stating it.';
}
