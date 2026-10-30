/**
 * Scripted-agent conformance scenarios for the platform-neutral core. They take the core module and a `Realm` and use
 * nothing else, so the same scenarios run under Node (test/core.test.mjs) and inside workerd (test/workerd-core.test.mjs).
 * A scripted agent plays the model: it drives the interpreter session directly, so no model is involved.
 */

const check = (condition, message) => { if (!condition) throw new Error(message); };

/** An agent that evaluates `code(session)` as its one eval and finishes with its value. */
const evaluating = code => async session => {
  const result = await session.applyAsync('eval', { code: code(session) });
  check(result.kind === 'ok' || result.kind === 'value', `eval -> ${result.kind}: ${result.text}`);
  check(session.finish(), 'the call did not finish');
};

const DOUBLE = '---\nargs:\n  n: number\nreturns: number\n---\nDouble n.\n';

export function scenarios(core, realm) {
  const evals = { count: 0 };
  const counting = { call: (body, parameters, args, options) => { evals.count++; return realm.call(body, parameters, args, options); } };
  const platform = core.workerdPlatform({ realm: counting });
  const runtime = options => core.createNatlangRuntime({ platform, calls: false, ...options });

  return {
    async 'a named .nl call runs through the host realm'() {
      const double = core.loadVirtualNatlang({ 'double.nl': DOUBLE }, 'double.nl', '/project', platform);
      const before = evals.count;
      const value = await runtime({ agent: evaluating(() => 'return n * 2') }).run(() => double(21));
      check(value === 42, `expected 42, got ${JSON.stringify(value)}`);
      check(evals.count > before, 'the eval did not run in the host realm');
    },

    async 'an inline nl call compiles with compileVirtualProject and runs'() {
      const compiled = core.compileVirtualProject({ files: { 'main.ts':
        "import { nl } from '@natlang/browser';\nexport async function main(a: number): Promise<number> { return await nl<number>`Double a.`(a); }\n" } },
        core, { platform });
      check(compiled.ok, `the project did not compile: ${JSON.stringify(compiled.diagnostics)}`);
      const app = compiled.require('main.ts');
      const value = await runtime({ agent: evaluating(session => `return ${Object.keys(session.lam.args)[0]} * 2`) }).run(() => app.main(4));
      check(value === 8, `expected 8, got ${JSON.stringify(value)}`);
    },

    async 'services are traced as effects'() {
      const seen = [];
      const ledger = { record: async n => { seen.push(n); return seen.length; } };
      const log = core.loadVirtualNatlang({ 'log.nl': '---\nargs:\n  n: number\nreturns: number\n---\nRecord n in the ledger.\n' }, 'log.nl', '/project', platform);
      const traces = [];
      const value = await runtime({ agent: evaluating(() => 'await ledger.record(n); return n'), services: { ledger }, trace: trace => traces.push(trace) })
        .run(() => log(7));
      check(value === 7 && seen.join() === '7', `the service was not called once with 7: ${seen}`);
      const effects = traces.flatMap(trace => trace.events).filter(event => event.kind === 'effect' && event.phase === 'completed').map(event => event.capability);
      check(effects.join() === 'ledger.record', `expected one traced ledger.record effect, got ${JSON.stringify(effects)}`);
    },

    async 'a failed call surfaces as NatlangCallError'() {
      const double = core.loadVirtualNatlang({ 'double.nl': DOUBLE }, 'double.nl', '/project', platform);
      const agent = async session => session.apply('return_result', { status: 'blocked', reason: 'cannot double this' }).text;
      let error;
      try { await runtime({ agent }).run(() => double(1)); } catch (caught) { error = caught; }
      check(error instanceof core.NatlangCallError, `expected NatlangCallError, got ${error}`);
      check(/cannot double this/.test(error.detail), `the reason is lost: ${error.detail}`);
      check(typeof error.outcome === 'string' && error.outcome !== 'done', `outcome ${error.outcome}`);
    },

    async 'aborting the task cancels a running call'() {
      const double = core.loadVirtualNatlang({ 'double.nl': DOUBLE }, 'double.nl', '/project', platform);
      let started;
      const running = new Promise(resolve => { started = resolve; });
      let observed = false;
      const agent = async session => {
        started();
        await new Promise(resolve => session.runtime.signal.addEventListener('abort', resolve, { once: true }));
        observed = true;
        throw session.runtime.signal.reason ?? new Error('aborted');
      };
      const controller = new AbortController();
      const call = runtime({ agent }).run(() => double(1), { signal: controller.signal });
      const outcome = call.then(() => 'completed', error => error);
      await running;
      controller.abort(new Error('stop requested'));
      const result = await outcome;
      check(result !== 'completed', 'the aborted call completed');
      check(observed, 'the call never saw the abort');
    },
  };
}
