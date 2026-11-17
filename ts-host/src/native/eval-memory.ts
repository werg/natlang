/**
 * The memory bound of one evaluation of model-written code. Eval code runs in its host's own heap (a `node:vm` context
 * shares the process isolate with every other episode the process serves), so an evaluation whose data explodes (an
 * unpruned search, a doubling array) used to grow the heap until the host was killed with every co-hosted episode
 * (teacher generation 2026-11-16: Sokoban BFS/A* solvers in eval OOM-killed the collector four times).
 *
 * The bound is per evaluation and crisp: compiled eval code calls the guard at the top of every loop iteration and
 * function body (`instrumentMemoryChecks`); every `CHECK_EVERY` calls the guard reads the heap. When the heap has grown
 * more than the bound since the evaluation started, it runs a full collection so that garbage is not counted, and if
 * the growth that survives is still over the bound the evaluation throws the teaching error. Once tripped, every later
 * check of that evaluation throws again, so code that catches the error cannot keep growing. Only the evaluation that
 * grows fails: its own code is the one that reaches the check.
 */
import ts from 'typescript';

/** The name compiled eval code calls; a realm that binds it runs instrumented code. */
export const MEMORY_CHECK = '__natlang_memory';
/**
 * Default bound in MB. Data (2026-11-16, 20,582 `invocation_end` records of shared collector processes serving up to
 * 8 RL/teacher episodes each): the whole process heap had a median of 147 MB and p99 323 MB, i.e. episodes need tens of
 * MB. 256 MB is an order of magnitude above a legitimate evaluation, and 8 evaluations at the bound still fit a 4 GB
 * collector heap. Setting: `NATLANG_EVAL_MEMORY_MB` (0 turns the bound off) or the environment's `memoryLimitMb`. The same
 * bound covers callable-folder module code (`segmentGuard`), the folder `python` tool (WASM memory growth per cell,
 * native/folder-python.ts) and the folder `bash` tool (just-bash byte limits and a write budget, native/folder-shell.ts).
 * Browser evals stay unbounded: a page has no synchronous, precise heap reading (`performance.memory` is coarse and
 * Chrome-only, `measureUserAgentSpecificMemory` is asynchronous), and a runaway there ends only the user's own tab.
 */
export const DEFAULT_EVAL_MEMORY_MB = 256;
/** Guard calls between heap readings: a reading costs ~0.35 us on Node, so a check costs a few ns per call. */
const CHECK_EVERY = 128;
const MB = 1024 * 1024;

/** How a platform reads its heap: bytes in use, and a full collection (absent: growth is judged without one). */
export type HeapProbe = { used(): number; collect?(): void };

/** The bound in MB: `NATLANG_EVAL_MEMORY_MB` (0 turns it off), else `DEFAULT_EVAL_MEMORY_MB`. One setting for eval, folder
 * Python, folder bash and callable-folder module code. */
export function evalMemoryLimitMb(env: Record<string, string | undefined> =
    (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env ?? {}): number {
  const text = env.NATLANG_EVAL_MEMORY_MB?.trim();
  if (!text) return DEFAULT_EVAL_MEMORY_MB;
  const value = Number(text);
  if (!Number.isFinite(value) || value < 0) throw new RangeError(`NATLANG_EVAL_MEMORY_MB must be a number of MB >= 0 (0: unbounded), got ${text}`);
  return value;
}

/** The one-sentence teaching error of a run that exceeded its bound: `eval`, `python`, `bash`, callable-folder code. */
export function memoryLimitMessage(subject: string, unit: string, limitMb: number): string {
  return `${subject} stopped: its data grew past the ${limitMb} MB memory limit of one ${unit}; use a bounded or more ` +
    'memory-efficient approach (prune the search, keep compact state, or process the data in parts)';
}
export const evalMemoryMessage = (limitMb: number) => memoryLimitMessage('eval', 'eval', limitMb);
export const moduleMemoryMessage = (limitMb: number) =>
  memoryLimitMessage('callable-folder code', 'synchronous run of callable-folder code', limitMb);

/** The guard of one evaluation: call it from the evaluation's code; it throws once the evaluation's growth is over. */
export function memoryGuard(limitMb: number, probe: HeapProbe, message = evalMemoryMessage(limitMb)): () => void {
  const limit = limitMb * MB, base = probe.used();
  let ticks = 0, next = base + limit, tripped = false;
  const fail = () => { tripped = true; throw new RangeError(message); };
  return () => {
    if (tripped) fail();
    if (++ticks < CHECK_EVERY) return;
    ticks = 0;
    if (probe.used() <= next) return;
    if (!probe.collect) fail();
    probe.collect!();
    const after = probe.used();
    if (after - base > limit) fail();
    // Garbage, not growth: allow at least half the bound of fresh allocation before collecting again.
    next = Math.max(base + limit, after + limit / 2);
  };
}

/**
 * The guard of code that runs on behalf of many callers at unknown times (callable-folder module functions): each
 * synchronous run (up to the next microtask checkpoint) is one bounded run with its own base. Growth spread over
 * several runs (across awaits) is not summed.
 */
export function segmentGuard(limitMb: number, probe: HeapProbe, message = moduleMemoryMessage(limitMb)): () => void {
  let current: (() => void) | undefined;
  return () => {
    if (!current) {
      current = memoryGuard(limitMb, probe, message);
      queueMicrotask(() => { current = undefined; });
    }
    current();
  };
}

/**
 * `source` with a guard call at the top of every loop iteration and of every function body except constructors
 * (their loops are instrumented). Edits are insertions on the lines they belong to, so line numbers do not move.
 * Source that does not parse is returned as it is: compilation reports its errors.
 */
export function instrumentMemoryChecks(source: string, check = MEMORY_CHECK, kind: 'ts' | 'js' = 'ts'): string {
  const file = ts.createSourceFile(kind === 'ts' ? 'natlang-eval.ts' : 'natlang-module.js', source, ts.ScriptTarget.Latest, true,
    kind === 'ts' ? ts.ScriptKind.TS : ts.ScriptKind.JS);
  if ((file as unknown as { parseDiagnostics?: unknown[] }).parseDiagnostics?.length) return source;
  const call = `${check}();`;
  // [position, order at that position (ends before starts in the text), text]
  const edits: [number, number, string][] = [];
  const atBlockStart = (block: ts.Block) => edits.push([block.getStart(file) + 1, 1, call]);
  const statementBody = (body: ts.Statement) => {
    if (ts.isBlock(body)) return atBlockStart(body);
    edits.push([body.getStart(file), 1, `{${call}`], [body.getEnd(), 0, '}']);
  };
  const visit = (node: ts.Node): void => {
    if (ts.isIterationStatement(node, false)) statementBody(node.statement);
    else if ((ts.isFunctionDeclaration(node) || ts.isFunctionExpression(node) || ts.isMethodDeclaration(node) ||
      ts.isGetAccessorDeclaration(node) || ts.isSetAccessorDeclaration(node)) && node.body) atBlockStart(node.body);
    else if (ts.isArrowFunction(node)) {
      if (ts.isBlock(node.body)) atBlockStart(node.body);
      else edits.push([node.body.getStart(file), 1, `(${check}(), `], [node.body.getEnd(), 0, ')']);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  // Apply from the end; at one position the start is applied first so that an end lands before it in the text.
  edits.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
  for (const [position, , text] of edits) source = source.slice(0, position) + text + source.slice(position);
  return source;
}
