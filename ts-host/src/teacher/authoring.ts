/**
 * Acceptance for TypeScript authoring tasks: the project folder a directory reducer edited is mounted as a
 * callable folder and its authored export is called on test inputs. Its `nl` children are answered by an
 * oracle keyed by fragments of the child's opening (its argument values), since the author's instruction
 * wording is not known in advance; an unmatched child fails, which fails the check.
 */
import { isDeepStrictEqual } from 'node:util';
import { createNatlangRuntime } from '../runtime/node.js';
import { loadVirtualNatlang } from '../runtime/virtual-project.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { openingText, type Message } from './opening.js';

export type AuthoringSpec = {
  /** The authored module, a path in the folder (for example "app.ts"), and the export to call. */
  module: string;
  export: string;
  /** Calls to make and the results they must produce. */
  runs: { args: unknown[]; expected: unknown }[];
  /** Answers for the export's natural-language children: every fragment must appear in the child's opening. */
  oracle?: { match: string[]; value: unknown }[];
  /** What the authored source must use. */
  requires?: { nl?: boolean; iterateOn?: boolean; calls?: string[]; noNl?: boolean };
};

export type AuthoringResult = { ok: boolean; problems: string[]; results: unknown[] };

const HARNESS = 'authoring_check';

/** Run the authored export on every test input. */
export async function checkAuthoring(files: Record<string, string>, spec: AuthoringSpec): Promise<AuthoringResult> {
  const problems: string[] = [];
  const source = files[spec.module];
  if (source === undefined) return { ok: false, problems: [`${spec.module} does not exist`], results: [] };
  if (spec.requires?.nl && !/\bnl\s*(?:<[^`]*>)?\s*`/.test(source)) problems.push(`${spec.module} creates no nl function`);
  if (spec.requires?.iterateOn && !/\biterateOn\b/.test(source)) problems.push(`${spec.module} does not use iterateOn`);
  if (spec.requires?.noNl && /\bnl\s*(?:<[^`]*>)?\s*`/.test(source)) problems.push(`${spec.module} creates an nl function where an existing one fits`);
  for (const name of spec.requires?.calls ?? []) if (!new RegExp(`\\b${name}\\s*\\(`).test(source)) problems.push(`${spec.module} does not call ${name}`);
  // The project's natlang and TypeScript files become the harness's callable folder.
  const mounted: Record<string, string> = {};
  for (const [path, text] of Object.entries(files)) if (/\.(?:ts|nl)$/.test(path)) mounted[`${HARNESS}/${path}`] = text;
  const moduleName = spec.module.replace(/\.ts$/, '').split('/').join('.');
  mounted[`${HARNESS}.nl`] = `---\nargs:\n  payload: string\nreturns: string\n---\nRun the check.\n`;
  const results: unknown[] = [];
  for (const [index, run] of spec.runs.entries()) {
    let staged = false;
    const model = async (request: ModelTurnRequest): Promise<ModelTurn> => {
      const opening = String((request.messages[1] as { content?: unknown } | undefined)?.content ?? '');
      if (opening.startsWith(`You are inside this call: ${HARNESS}(`)) {
        // After the eval staged its result, a reply without a tool call returns it; a failed eval ends the call.
        if (staged) {
          const last = String((request.messages.at(-1) as { content?: unknown } | undefined)?.content ?? '');
          return /\nStaged /.test(`\n${last}`) || last.includes('Staged ') ? { text: 'done' } :
            { calls: [['return_result', { status: 'failed', reason: `The authored export failed: ${last.slice(0, 400)}` }]] };
        }
        staged = true;
        return { calls: [['eval', { code: `return JSON.stringify(await ${moduleName}.${spec.export}(...JSON.parse(payload)) ?? null);` }]] };
      }
      const shown = openingText(request.messages as Message[]);
      const answer = spec.oracle?.find(item => item.match.every(fragment => shown.includes(fragment)));
      return answer ? { calls: [['return_result', { status: 'success', value: answer.value }]] } :
        { calls: [['return_result', { status: 'failed', reason: 'The authoring oracle has no answer for this call.' }]] };
    };
    const runtime = createNatlangRuntime({ model: model as never, seed: { mode: 'backend' } });
    try {
      const fn = loadVirtualNatlang(mounted, `${HARNESS}.nl`);
      const text = await runtime.run(() => (fn as unknown as (payload: string) => Promise<string>)(JSON.stringify(run.args)));
      const value = JSON.parse(String(text));
      results.push(value);
      if (!isDeepStrictEqual(value, run.expected)) problems.push(`run ${index}: got ${JSON.stringify(value)}, expected ${JSON.stringify(run.expected)}`);
    } catch (error) {
      results.push(null);
      problems.push(`run ${index}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  return { ok: !problems.length, problems, results };
}
