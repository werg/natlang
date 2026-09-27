/** A reproducible rubric judge for `judged` program oracles. */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export type JudgeTurn = (request: ModelTurnRequest) => Promise<ModelTurn>;
export type JudgeInput = { actual: unknown; expected: unknown; rubric: string };
export type JudgeResult = { accepted: boolean; verdict: string };

const GRADE_TOOL = { type: 'function', function: { name: 'grade',
  description: 'Grade the candidate against the reference and rubric, with a concise evidence-based verdict.',
  parameters: { type: 'object', properties: {
    accepted: { type: 'boolean' }, verdict: { type: 'string' },
  }, required: ['accepted', 'verdict'], additionalProperties: false } } };

export function modelOracleJudge(turn: JudgeTurn, options: { maxTokens?: number } = {}) {
  return async ({ actual, expected, rubric }: JudgeInput): Promise<JudgeResult> => {
    const request: ModelTurnRequest = { messages: [
      { role: 'system', content: 'You are an independent grading model. Apply the supplied rubric to the candidate ' +
        'against the reference. Call grade once. Accept only when the candidate meets the rubric. Explain the decisive evidence.' },
      { role: 'user', content: JSON.stringify({ rubric, reference: expected, candidate: actual }) },
    ], tools: [GRADE_TOOL], tool_choice: 'required', temperature: 0, seed: null, max_tokens: options.maxTokens ?? 512 };
    const response = await turn(request);
    const found = response.calls?.find(([name]) => name === 'grade')?.[1];
    const value: unknown = found ?? (response.text ? JSON.parse(response.text) : undefined);
    if (!value || typeof value !== 'object' || Array.isArray(value) ||
      typeof (value as Record<string, unknown>).accepted !== 'boolean' ||
      typeof (value as Record<string, unknown>).verdict !== 'string' ||
      !(value as { verdict: string }).verdict.trim())
      throw new Error('judge did not return a valid grade(accepted, verdict)');
    return { accepted: (value as JudgeResult).accepted, verdict: (value as JudgeResult).verdict.trim() };
  };
}
