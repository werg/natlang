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
 * collector heap. Setting: `NATLANG_EVAL_MEMORY_MB` (0 turns the bound off) or the environment's `memoryLimitMb`.
 */
export const DEFAULT_EVAL_MEMORY_MB = 256;
/** Guard calls between heap readings: a reading costs ~0.35 us on Node, so a check costs a few ns per call. */
const CHECK_EVERY = 128;
const MB = 1024 * 1024;

/** How a platform reads its heap: bytes in use, and a full collection (absent: growth is judged without one). */
export type HeapProbe = { used(): number; collect?(): void };

/** The one-sentence teaching error of an evaluation that exceeded its bound. */
export function evalMemoryMessage(limitMb: number): string {
  return `eval stopped: its data grew past the ${limitMb} MB memory limit of one eval; use a bounded or more ` +
    'memory-efficient approach (prune the search, keep compact state, or process the data in parts)';
}

/** The guard of one evaluation: call it from the evaluation's code; it throws once the evaluation's growth is over. */
export function memoryGuard(limitMb: number, probe: HeapProbe): () => void {
  const limit = limitMb * MB, base = probe.used();
  let ticks = 0, next = base + limit, tripped = false;
  const fail = () => { tripped = true; throw new RangeError(evalMemoryMessage(limitMb)); };
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
 * `source` with a guard call at the top of every loop iteration and of every function body except constructors
 * (their loops are instrumented). Edits are insertions on the lines they belong to, so line numbers do not move.
 * Source that does not parse is returned as it is: compilation reports its errors.
 */
export function instrumentMemoryChecks(source: string, check = MEMORY_CHECK): string {
  const file = ts.createSourceFile('natlang-eval.ts', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
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
