/**
 * Guard dispatch and the hand-off to the agent (§6.4, §6.5). The kernel asks which case admits a call, runs it, and
 * when it throws gives the agent one note saying what the case already did.
 */
import type { LoadedCase, LoadedCompilation } from './compilations.js';
import type { EffectRecord } from './types.js';

/** A case declines an input its guard admitted; the call goes to the agent as if no case had admitted it. */
export class Deopt extends Error {
  constructor(reason = 'the case does not handle this input') { super(reason); this.name = 'Deopt'; }
}
export const isDeopt = (error: unknown): boolean => error instanceof Error && error.name === 'Deopt';

/** The first active case whose guard admits `args` (in `on` mode), and every shadow case whose guard admits them. */
export function admit(compilation: LoadedCompilation, args: Record<string, unknown>, mode: 'on' | 'shadow'):
  { active?: LoadedCase; shadows: LoadedCase[]; guardErrors: { hash: string; error: string }[] } {
  const shadows: LoadedCase[] = [], guardErrors: { hash: string; error: string }[] = [];
  let active: LoadedCase | undefined;
  for (const item of compilation.cases) {
    if (item.tier !== 'active' && item.tier !== 'shadow') continue;
    let admitted = false;
    try {
      const verdict = item.when(args);
      if (verdict && typeof (verdict as PromiseLike<unknown>).then === 'function') {
        (verdict as Promise<unknown>).catch(() => {});
        guardErrors.push({ hash: item.hash, error: 'a guard must answer synchronously; async guards never admit' });
      } else admitted = verdict === true;
    } catch (error) { guardErrors.push({ hash: item.hash, error: error instanceof Error ? error.message : String(error) }); }
    if (!admitted) continue;
    if (item.tier === 'active' && mode === 'on' && !active) active = item;
    else if (item.tier === 'shadow' || (item.tier === 'active' && mode === 'shadow')) shadows.push(item);
  }
  return { active, shadows, guardErrors };
}

const short = (value: unknown, limit = 300): string => {
  let text: string;
  try { text = typeof value === 'string' ? value : JSON.stringify(value) ?? String(value); } catch { text = String(value); }
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
};

/**
 * The note the agent's opening carries after a case stopped: the error, the effects that already happened, the files
 * the case changed and the calls it finished. Shown only on a hand-off; one paragraph, no new names.
 */
export function handoffNote(input: { error: string; effects: { effect: EffectRecord; args?: unknown; result?: unknown }[];
  files: { path: string; kind: string }[]; calls: { name: string; outcome: string }[] }): string {
  const done = [
    ...input.effects.map(({ effect, args, result }) => `${effect.service}.${effect.method}(${short(args ?? '', 160).replace(/^\[|\]$/g, '')})` +
      (effect.error ? ` failed: ${short(effect.error, 160)}` : ` returned ${short(result, 160)}`)),
    ...input.calls.map(call => `${call.name}(...) ${call.outcome === 'done' ? 'finished' : `ended: ${call.outcome}`}`),
    ...input.files.map(file => `${file.kind} ${file.path}`)];
  return `A compiled fast path for this call started and stopped: ${short(input.error, 400)}. ` +
    (done.length ? `Before it stopped it already did this, and these effects have happened:\n- ${done.join('\n- ')}\n` +
      'Continue from this state to complete the call. Do not repeat a finished effect unless the instructions need it again.' :
      'It had done nothing yet; carry out the call as usual.');
}
