/** Test helpers: build callable-folder records and interpreter sessions without files on disk. */
import { NodeNativeRuntime } from '../../dist/node-runtime.js';
import { modelTurnsSoFar } from '../../dist/native/agent.js';
import { NativeSession } from '../../dist/native/runtime.js';
import { TypeEnv } from '../../dist/native/types.js';
import { buildPending } from '../../dist/native/values.js';
import { parseModule, parseNatlang, PATH_ONLY } from '../../dist/runtime/loader.js';
import YAML from 'yaml';

/** A TypeScript module item (callable-folder `.ts` file). */
export function ts(name, text, children = {}, types = {}) {
  const record = parseModule(`${name}.ts`, text, types, PATH_ONLY);
  record.codebase = children;
  return record;
}

/** A natural-language function item (`.nl` file). */
export function nl(name, { args = {}, returns, instructions, types = {}, kind, description } = {}, children = {}) {
  const meta = { ...(description ? { description } : {}), args, returns, ...(kind ? { kind } : {}) };
  const record = parseNatlang(`${name}.nl`, `---\n${YAML.stringify(meta)}---\n${instructions}\n`, types, PATH_ONLY);
  record.codebase = children;
  return record;
}

/** A standalone interpreter bound to the kernel and its own task. */
export function interpreter(options = {}) { return new NodeNativeRuntime(options); }

/** A lambda node from a `$lambda` body (instructions, type, args, codebase, types, subtype). */
export function lambda(body) { return buildPending({ $lambda: body }); }

/** An interpreter session over one lambda. */
export function session(body, options = {}) {
  const lam = lambda(body);
  const runtime = interpreter(options);
  const env = new TypeEnv().child(lam.types);
  return { lam, runtime, session: new NativeSession(runtime, lam, env) };
}

export { NativeSession, TypeEnv };

/**
 * A scripted model: `respond(opening)` returns eval code for a natlang invocation (or null to report an
 * error). The driver evaluates it, then replies done.
 */
export function scriptedModel(respond) {
  const openings = [];
  const driver = async ({ messages }) => {
    // What the model sees first: the call and instructions, then the runtime's pre-filled scope code and its results.
    const opening = [String(messages[1].content), ...messages.slice(2).flatMap(message =>
      message.role === 'assistant' ? (message.tool_calls ?? []).filter(call => String(call.id).startsWith('scope_'))
        .map(call => JSON.parse(call.function.arguments).code ?? '') :
      message.role === 'tool' && String(message.tool_call_id).startsWith('scope_') ? [String(message.content)] : [])].join('\n');
    const last = messages.at(-1);
    if (modelTurnsSoFar(messages) === 0) {
      openings.push(opening);
      const code = await respond(opening);
      if (code === null) return { calls: [['failed', { message: 'The scripted model has no answer for this task.' }]] };
      return { calls: [['eval', { code }]] };
    }
    if (last.role === 'tool' && /^(?:rejected|error)|\nerror|Nothing else from this eval was kept/.test(String(last.content)))
      return { calls: [['failed', { message: `Scripted eval failed: ${String(last.content).slice(0, 300)}` }]] };
    return { text: 'done' };
  };
  return { driver, openings };
}
