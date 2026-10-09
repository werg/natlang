/**
 * What the model sees for each built-in natlang function (the progress judge, the stopping-condition note, the
 * behavior-comparison judge, the digest prompt and the game policy), captured through public entry points with a
 * recording driver. `test/fixtures/builtin-prompts.json` was generated from this file at the commit where these
 * functions were still string literals in TypeScript; builtin-programs.test.mjs asserts the bytes are unchanged.
 */
import { createNatlangRuntime, iterateOn } from '../../dist/runtime/node.js';
import { defineNatlang } from '../../dist/runtime/callable.js';
import { promptPieces } from '../../dist/native/system-prompts.js';
import { judge } from '../../dist/calls/judge.js';
import { GAME_POLICY_SOURCE } from '../../dist/self-play/policy.js';
import { loadVirtualNatlang } from '../../dist/runtime/virtual-project.js';

/** Per-run identifiers are not part of what the model reads, nor is `natlang_host_generated`, internal provenance that
 * the chat-completions driver strips before the wire (model/chat-completion.ts). */
function normalize(value) {
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object')
    return Object.fromEntries(Object.entries(value).filter(([key]) => key !== 'natlang_host_generated')
      .map(([key, item]) => [key, key === 'invocation_id' ? '<invocation>' : normalize(item)]));
  return typeof value === 'string' ? value.replace(/iter-[a-z0-9]+/g, 'iter-<id>') : value;
}

/** A driver that records every request and answers each with `answer(request, index)`. */
function recorder(answer) {
  const requests = [];
  const driver = async request => { requests.push(normalize(JSON.parse(JSON.stringify(request)))); return answer(request, requests.length - 1); };
  return { requests, driver };
}
const ok = value => ({ calls: [['return_result', { status: 'success', value }]] });

export async function renderBuiltinPrompts() {
  const out = {};
  // The progress judge: the bootstrap review of a long default-judged loop.
  {
    const { requests, driver } = recorder(() => ok({ verdict: 'continue', reason: 'still improving' }));
    const runtime = createNatlangRuntime({ model: driver });
    await runtime.run(() => iterateOn(n => n + 1, 0).withLimit({ maxSteps: 11 }).until(n => n >= 11));
    out.progressJudge = requests;
  }
  // The stopping condition's note: the initial state, then a state unchanged over steps.
  {
    const { requests, driver } = recorder(() => ok(false));
    const runtime = createNatlangRuntime({ model: driver });
    const stop = defineNatlang('---\nargs:\n  state: number\nreturns: boolean\n---\nThe state is at least 1.\n', { name: 'stop' });
    await runtime.run(() => iterateOn(n => n, 0).checkProgress('off').withLimit({ maxSteps: 2 }).until(stop)).catch(() => {});
    const changing = { requests: requests.length };
    await runtime.run(() => iterateOn(n => n + 1, 0).checkProgress('off').withLimit({ maxSteps: 2 }).until(stop)).catch(() => {});
    out.stoppingCondition = { requests, firstRunRequests: changing.requests };
  }
  // The behavior-comparison judge, with both orders.
  {
    const { requests, driver } = recorder(() => ok('first'));
    const runtime = createNatlangRuntime({ model: driver });
    const behavior = (value, effects = []) => ({ value, effects });
    for (const seed of ['a', 'b', 'c', 'd'])
      await judge(runtime, { instructions: 'Add the numbers.', signature: 'add(a: number, b: number): number', inputs: { a: 1, b: 2 },
        reference: behavior(3, [{ service: 'log', method: 'write', args: ['x'] }]), candidate: behavior(4), seed });
    out.compareBehaviors = requests;
  }
  out.digest = promptPieces().find(piece => piece.id === 'digest').text;
  out.gamePolicySource = GAME_POLICY_SOURCE;
  // The game policy as a model call.
  {
    const { requests, driver } = recorder(() => ok('{"action":"pass"}'));
    const runtime = createNatlangRuntime({ model: driver });
    const play = loadVirtualNatlang({ 'play.nl': GAME_POLICY_SOURCE }, 'play.nl');
    await runtime.run(() => play('{"seat":"a","rules":"r","observation":"o","actions":["pass"]}'));
    out.gamePolicy = requests;
  }
  out.promptPieces = promptPieces();
  return out;
}
