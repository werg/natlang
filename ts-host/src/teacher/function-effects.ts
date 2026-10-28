/**
 * Function-effect oracle: a case whose contract is a change to one of the program's own functions (repair a helper
 * that breaks its documented contract, or leave a correct one alone) is judged on that function as the run left it,
 * not only on the returned value. 2026-10-26 audit: a held contract_diagnosis case accepted a hard-coded total
 * returned without the required repair. The function's final source is compiled and called on hidden checks in a
 * fresh V8 context (no require, no host globals, bounded time); the returned value must agree with what the final
 * function computes over the case's inputs.
 */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import vm from 'node:vm';
import ts from 'typescript';

export type FunctionEffectSpec = { schema: 'natlang.function-effect-oracle/1';
  /** The function's name in the program's codebase (a module with a default export). */
  function: string;
  /** required: the run must change the function (it breaks its contract); forbidden: it must leave it unchanged. */
  change: 'required' | 'forbidden';
  /** Hidden calls of the final function and their contract results. */
  runs: { args: unknown[]; expected: unknown }[];
  /** The returned value is this combination of the final function over one input list. */
  value_from?: { input: string; combine: 'sum' } };
export type FunctionEffectResult = { ok: boolean; problems: string[]; changed: boolean;
  original_sha256: string | null; final_sha256: string | null; comparison_version: 'function-effects/1' };

const CALL_TIMEOUT_MS = 1000;
const sha = (text: string) => createHash('sha256').update(text).digest('hex');

/** Compile a module's TypeScript source in a fresh context and return its default export (or an error message). */
function loadDefault(source: string): { fn?: (...args: unknown[]) => unknown; context?: vm.Context; error?: string } {
  try {
    const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
    const context = vm.createContext({ module: { exports: {} as Record<string, unknown> } });
    vm.runInContext(`(function (module, exports) {\n${js}\n})(module, module.exports);`, context, { timeout: CALL_TIMEOUT_MS });
    const exports = (context.module as { exports: Record<string, unknown> }).exports;
    const fn = exports.default ?? (typeof exports === 'function' ? exports : undefined);
    if (typeof fn !== 'function') return { error: 'has no default export function' };
    return { fn: fn as (...args: unknown[]) => unknown, context };
  } catch (error) { return { error: `does not load: ${error instanceof Error ? error.message : String(error)}` }; }
}

/** Call the loaded function inside its own context so the time bound covers the call. */
function call(context: vm.Context, fn: (...args: unknown[]) => unknown, args: unknown[]): { value?: unknown; error?: string } {
  try {
    context.__fn = fn; context.__args = JSON.stringify(args);
    const value = vm.runInContext('__fn(...JSON.parse(__args))', context, { timeout: CALL_TIMEOUT_MS });
    // Results cross the context boundary as JSON values (numbers, strings, plain records).
    return { value: value === undefined ? undefined : JSON.parse(JSON.stringify(value)) };
  } catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
}

export function checkFunctionEffects(spec: FunctionEffectSpec, original: string | undefined, final: string | undefined,
  value: unknown, inputs: Record<string, unknown>): FunctionEffectResult {
  const problems: string[] = [];
  const changed = original !== undefined && final !== undefined && final !== original;
  const result = (): FunctionEffectResult => ({ ok: !problems.length, problems, changed,
    original_sha256: original === undefined ? null : sha(original), final_sha256: final === undefined ? null : sha(final),
    comparison_version: 'function-effects/1' });
  if (original === undefined || final === undefined) { problems.push(`${spec.function} is not a function of this program`); return result(); }
  if (spec.change === 'required' && !changed) problems.push(`${spec.function} was not repaired`);
  if (spec.change === 'forbidden' && changed) problems.push(`${spec.function} meets its contract but was changed`);
  const loaded = loadDefault(final);
  if (!loaded.fn || !loaded.context) { problems.push(`${spec.function} ${loaded.error}`); return result(); }
  for (const [i, run] of spec.runs.entries()) {
    const got = call(loaded.context, loaded.fn, run.args);
    if (got.error !== undefined) problems.push(`hidden check ${i + 1}: ${spec.function} threw ${got.error}`);
    else if (!isDeepStrictEqual(got.value, run.expected)) problems.push(`hidden check ${i + 1}: ${spec.function} returned ${JSON.stringify(got.value)}, not ${JSON.stringify(run.expected)}`);
    // A function that does not finish within the bound is wrong; stop rather than wait on every remaining call.
    if (got.error !== undefined && /timed out/i.test(got.error)) return result();
  }
  if (spec.value_from) {
    const items = inputs[spec.value_from.input];
    if (!Array.isArray(items)) problems.push(`input ${spec.value_from.input} is not a list`);
    else {
      let total = 0, failed = false;
      for (const item of items) {
        const got = call(loaded.context, loaded.fn, [item]);
        if (got.error !== undefined || typeof got.value !== 'number') { failed = true; break; }
        total += got.value;
      }
      if (failed) problems.push(`${spec.function} does not compute a number for every item of ${spec.value_from.input}`);
      else if (!isDeepStrictEqual(value, total)) problems.push(`the returned value ${JSON.stringify(value)} is not the sum of ${spec.function} over ${spec.value_from.input} (${total})`);
    }
  }
  return result();
}
