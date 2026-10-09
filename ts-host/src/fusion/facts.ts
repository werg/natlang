/**
 * The crisp fact service of fused pipelines (plans/FUSED_PIPELINES.md). For a project it returns the candidate edges of
 * the call graph, a producer function whose result reaches a consumer function, with the exact facts a planner needs:
 * the functions, the declared type, each side's model name, and every reader of the value. It decides nothing.
 *
 * Two kinds of orchestrator are read:
 *
 * - A natural-language function with stages in its callable folder. Its instructions name the stages as calls
 *   (`a = cluster(item, open)`, `weigh(h, found)`). Each use of the variable a stage's result was bound to is classified
 *   by the call it sits in: a stage (the consumer or another one), a service, a crisp module, a field access or condition
 *   (the orchestrating model reads the value), a `Return` (the host reads it). Chains the prose only implies ("select,
 *   then liveness of the result") are reported as implicit edges with an unknown reader: nothing proves who reads them.
 * - Crisp TypeScript that calls named functions (`const plan = await database.plan(q); await database.execute(plan)`).
 *   Here every use of the variable is syntax, so the readers are exact.
 *
 * Nothing here is imported from compiler/eval-check.ts; it uses only the loader's records and the TypeScript parser.
 */
import ts from 'typescript';
import { loadNamedFunction, type ItemRecord, type NatlangRecord, type SourceFiles } from '../runtime/loader.js';

export const FUSION_FACTS_SCHEMA = 'natlang.fusion-facts/1';

/** Who reads a value besides (or as) the planned consumer. */
export type ReaderKind =
  /** The stage the value is passed to. */
  | 'consumer'
  /** Another stage or function the same value is passed to. */
  | 'other-call'
  /** Crisp callable-folder or host TypeScript. */
  | 'crisp-code'
  /** A host service (a verifier, an index, a store). */
  | 'service'
  /** The orchestrating model's own code or reasoning: a field access, a condition, a loop, a record it builds. */
  | 'eval'
  /** The value, or something built from it, is returned to the host. */
  | 'host-return'
  /** The value is logged, reported, displayed or shown to a user. */
  | 'trace-ui';

export type Reader = { kind: ReaderKind; name?: string; site: string; detail: string;
  /** False when the analysis could not see how the value is used and assumes the worst. */
  certain: boolean };

export type StageRef = { name: string; source: string; model: string | null; returns: string };
export type ConsumerRef = StageRef & { param: string | null; paramType: string | null };

export type EdgeFacts = {
  /** Stable ID: scope, producer source and consumer source. */
  id: string;
  /** The orchestrator the edge was found in (a `.nl` source, or `file.ts#function`). */
  scope: string;
  scopeKind: 'nl' | 'typescript';
  /** `producer -> consumer`, for people. */
  chain: string;
  producer: StageRef;
  consumer: ConsumerRef;
  /** The producer's declared result type; authors still see it. */
  type: string;
  /** How the value travels: `nested` (`B(A(x))`), `bound` (a variable), `implicit` (the prose sequences the calls), `typed` (only the declared types connect them). */
  flow: 'nested' | 'bound' | 'implicit' | 'typed';
  variable: string | null;
  /** The dialect both sides would speak: the runtime's default; the certificate states the exact one. */
  dialect: string;
  readers: Reader[];
  /** Call sites of the producer and the consumer in the scope. */
  producerSites: number;
  consumerSites: number;
  /** The producer already returns, or the consumer already takes, a Neuralese type. */
  alreadySoft: boolean;
  /** The result type has only a few values (a boolean, a set of literals): its text costs about one token. */
  finite: boolean;
  /** The sentence of the orchestrator's instructions the edge was read from (prose scopes). */
  excerpt?: string;
  /**
   * Present when the readers were read from recorded eval code in the call store instead of proven from the source:
   * the reader set is then observed, with its own run count and the store revision. Absent: the readers are proven.
   */
  observed?: ObservedEvidence;
};

export type FusionFacts = {
  schema: typeof FUSION_FACTS_SCHEMA;
  edges: EdgeFacts[];
  /** Orchestrators examined, and the stages each has. */
  scopes: { scope: string; kind: 'nl' | 'typescript'; stages: number }[];
  /** Files that could not be read, with the reason. */
  skipped: { source: string; reason: string }[];
};

const DEFAULT_DIALECT = 'DefaultDialect';
const SKIPPED = new Set(['node_modules', 'dist', '.git', '.natlang', 'build', 'vendor', 'test', 'tests', 'examples', 'bench']);

// --- Stage resolution ---------------------------------------------------------------------------------------------

type Stage = { path: string; record: NatlangRecord };

/** Every natural-language function below an orchestrator, by dotted path from its folder. */
function stagesOf(record: NatlangRecord): Stage[] {
  const out: Stage[] = [];
  const walk = (items: Record<string, ItemRecord>, prefix: string): void => {
    for (const [name, item] of Object.entries(items)) {
      const path = prefix ? `${prefix}.${name}` : name;
      if (item.kind === 'natlang') out.push({ path, record: item });
      walk(item.codebase, path);
    }
  };
  walk(record.codebase, '');
  return out;
}

const refOf = (record: NatlangRecord): StageRef => ({ name: record.name, source: record.source, model: record.model ?? null, returns: record.returns });

const isSoft = (type: string): boolean => /\bNeuralese\s*</.test(type);
const isFinite_ = (type: string): boolean => /^\s*(?:boolean|(?:"[^"]*"|'[^']*')(?:\s*\|\s*(?:"[^"]*"|'[^']*'))*)\s*$/.test(type);

function stageCandidates(stages: readonly Stage[], reference: string): Stage[] {
  const exact = stages.filter(stage => stage.path === reference);
  if (exact.length) return exact;
  // `opt.plan`, or a bare name that several folders share (the C, Python and Rust front ends).
  return stages.filter(stage => stage.path === reference || stage.path.endsWith(`.${reference}`));
}

// --- Prose orchestrators ------------------------------------------------------------------------------------------

const IDENT = '[A-Za-z_$][\\w$]*';
const BRACKETS: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
const RETURN_WORDS = /\b(?:return|returns|returned)\b/i;
const SHOW_WORDS = /\b(?:log|logs|report|reports|display|displays|show|shows|print|prints|record|records|note|notes)\b/i;

type CallSite = { start: number; open: number; close: number; reference: string; stages: Stage[] };

/** Where the parenthesis opened at `open` closes. */
function matchParen(text: string, open: number): number {
  let depth = 0;
  for (let at = open; at < text.length; at++) {
    const ch = text[at]!;
    if (ch === '(') depth++;
    else if (ch === ')' && --depth === 0) return at;
  }
  return text.length;
}

function callSites(text: string, stages: readonly Stage[]): CallSite[] {
  const sites: CallSite[] = [];
  const pattern = new RegExp(`(?<![\\w$.])(${IDENT}(?:\\.${IDENT})*)\\s*\\(`, 'g');
  for (const match of text.matchAll(pattern)) {
    const found = stageCandidates(stages, match[1]!);
    if (!found.length) continue;
    const open = match.index! + match[0].length - 1;
    sites.push({ start: match.index!, open, close: matchParen(text, open), reference: match[1]!, stages: found });
  }
  return sites;
}

/** The paragraph or list item (numbered or bulleted) of `at`. */
function statementAround(text: string, at: number): { start: number; end: number } {
  let start = at;
  while (start > 0) {
    const lineStart = text.lastIndexOf('\n', start - 1) + 1;
    if (/^\s*(?:\d+\.|-|\*)\s/.test(text.slice(lineStart, lineStart + 12)) || lineStart === 0 || /^\s*$/.test(text.slice(text.lastIndexOf('\n', lineStart - 2) + 1, lineStart - 1))) { start = lineStart; break; }
    start = lineStart - 1;
  }
  let end = text.length;
  const next = /\n(?:\s*\n|\s*(?:\d+\.|-|\*)\s)/g;
  next.lastIndex = at;
  const match = next.exec(text);
  if (match) end = match.index;
  return { start, end };
}

function sentenceAround(text: string, at: number): string {
  const { start, end } = statementAround(text, at);
  const body = text.slice(start, end);
  const local = at - start;
  const before = body.lastIndexOf('. ', local), after = body.indexOf('. ', local);
  return body.slice(before < 0 ? 0 : before + 2, after < 0 ? undefined : after + 1).replace(/\s+/g, ' ').trim();
}

/** A variable word is code-like at `at` when a short name cannot be an ordinary English word there. */
function codeLike(text: string, at: number, name: string): boolean {
  if (name.length > 2) return true;
  const before = text[at - 1] ?? ' ', after = text[at + name.length] ?? ' ';
  return /[(,{[:=`]/.test(before) || /[.),;\]}`[]/.test(after) || (before === ' ' && /\s*=/.test(text.slice(at + name.length, at + name.length + 3)));
}

type Context = { kind: 'call'; callee: string; position: number; open: number } | { kind: 'none' };

/** The innermost call whose argument list contains `at`, within the statement, and the argument position. */
function enclosingCall(text: string, at: number, floor: number): Context {
  const stack: { char: string; at: number; commas: number }[] = [];
  for (let index = floor; index < at; index++) {
    const ch = text[index]!;
    if ('([{'.includes(ch)) stack.push({ char: ch, at: index, commas: 0 });
    else if (')]}'.includes(ch)) { if (stack.at(-1)?.char === BRACKETS[ch]) stack.pop(); }
    else if (ch === ',' && stack.length) stack.at(-1)!.commas++;
  }
  for (let level = stack.length - 1; level >= 0; level--) {
    const open = stack[level]!;
    if (open.char !== '(') continue;
    const head = text.slice(Math.max(floor, open.at - 80), open.at).match(new RegExp(`(${IDENT}(?:\\.${IDENT})*)\\s*$`));
    if (head) return { kind: 'call', callee: head[1]!, position: open.commas, open: open.at };
  }
  return { kind: 'none' };
}

/** Parameter name and type at a position of a consumer call. */
function consumerParam(stage: Stage, position: number): { param: string | null; paramType: string | null } {
  const entries = Object.entries(stage.record.args);
  const entry = entries[position];
  if (!entry) return { param: null, paramType: null };
  return { param: entry[0].replace(/\?$/, ''), paramType: entry[1] };
}

type ProseUse = { kind: ReaderKind; name?: string; stage?: Stage; position?: number; detail: string; at: number };

function classifyUse(text: string, at: number, variable: string, stages: readonly Stage[], modules: ReadonlySet<string>,
    floor: number, end: number): ProseUse {
  const after = text.slice(at + variable.length, at + variable.length + 2);
  if (/^\.\w/.test(after) || after[0] === '[')
    return { kind: 'eval', detail: `field or index access ${variable}${after[0] === '[' ? '[…]' : '.…'}`, at };
  const context = enclosingCall(text, at, floor);
  const statement = text.slice(floor, end);
  if (context.kind === 'call') {
    const found = stageCandidates(stages, context.callee);
    if (found.length) return { kind: 'consumer', name: context.callee, stage: found[0]!, position: context.position,
      detail: `argument ${context.position + 1} of ${context.callee}`, at };
    const root = context.callee.split('.')[0]!;
    if (modules.has(root)) return { kind: 'crisp-code', name: context.callee, detail: `argument of crisp ${context.callee}`, at };
    if (context.callee.includes('.')) return { kind: 'service', name: context.callee, detail: `argument of ${context.callee}`, at };
  }
  const before = text.slice(floor, at);
  if (RETURN_WORDS.test(before)) return { kind: 'host-return', detail: 'part of what the function returns', at };
  if (SHOW_WORDS.test(statement)) return { kind: 'trace-ui', detail: 'mentioned in a log, report or display statement', at };
  return { kind: 'eval', detail: /\b(?:when|if|unless|otherwise|while)\b/i.test(statement) ? 'tested in a condition' : 'used by the orchestrating model', at };
}

const siteOf = (scope: string, text: string, at: number): string => `${scope}:${text.slice(0, at).split('\n').length}`;

/** The edges of one prose orchestrator. */
function proseEdges(record: NatlangRecord, stages: readonly Stage[]): EdgeFacts[] {
  const text = record.instructions, scope = record.source;
  const sites = callSites(text, stages);
  const modules = new Set(Object.entries(record.codebase).filter(([, item]) => item.kind === 'module').map(([name]) => name));
  const edges: EdgeFacts[] = [];
  const consumerCounts = new Map<string, number>();
  for (const site of sites) consumerCounts.set(site.stages[0]!.record.source, (consumerCounts.get(site.stages[0]!.record.source) ?? 0) + 1);
  const producerSites = (stage: Stage): number => sites.filter(item => item.stages.some(other => other.record.source === stage.record.source)).length;
  const seen = new Set<string>();
  for (const site of sites) {
    const { start: floor, end } = statementAround(text, site.start);
    const enclosing = enclosingCall(text, site.start, floor);
    const bind = new RegExp(`(?:^|[\\s(,;])(${IDENT}(?:\\.${IDENT})*)\\s*=\\s*(?:the result of\\s+)?$`).exec(text.slice(Math.max(floor, site.start - 60), site.start));
    const uses: (ProseUse & { uncertain?: boolean })[] = [];
    let variable: string | null = null, flow: EdgeFacts['flow'] = 'implicit';
    if (enclosing.kind === 'call' && stageCandidates(stages, enclosing.callee).length) {
      // B(A(x)): the value goes straight into the consumer and nowhere else.
      flow = 'nested';
      uses.push({ kind: 'consumer', name: enclosing.callee, stage: stageCandidates(stages, enclosing.callee)[0]!, position: enclosing.position,
        detail: `argument ${enclosing.position + 1} of ${enclosing.callee}`, at: site.start });
    } else if (bind) {
      const target = bind[1]!;
      if (target.includes('.')) {
        uses.push({ kind: 'eval', detail: `stored into ${target}`, at: site.start });
      } else {
        flow = 'bound'; variable = target;
        const word = new RegExp(`(?<![\\w$.])${target.replace(/\$/g, '\\$')}(?![\\w$])`, 'g');
        const bindAt = site.start - 0;
        word.lastIndex = site.close + 1;
        for (const match of text.matchAll(word)) {
          if (match.index! <= site.close || match.index! < bindAt) continue;
          if (!codeLike(text, match.index!, target)) continue;
          const area = statementAround(text, match.index!);
          // A later assignment to the name starts a new value.
          if (new RegExp(`^\\s*=(?!=)`).test(text.slice(match.index! + target.length, match.index! + target.length + 4)) &&
              !text.slice(area.start, match.index!).includes('(')) break;
          uses.push(classifyUse(text, match.index!, target, stages, modules, area.start, area.end));
        }
      }
    } else if (RETURN_WORDS.test(text.slice(floor, site.start))) {
      uses.push({ kind: 'host-return', detail: 'returned directly', at: site.start });
    } else {
      uses.push({ kind: 'eval', detail: 'the prose does not say where the value goes', at: site.start, uncertain: true });
    }
    const consumers = uses.filter(use => use.kind === 'consumer');
    // An implicit chain: "A(x), then B of the result".
    let implicitNext = flow === 'implicit' || !consumers.length ? implicitConsumers(text, site, sites) : [];
    if (!consumers.length && !implicitNext.length) implicitNext = typedConsumers(text, site, sites);
    for (const consumer of [...consumers, ...implicitNext.map(next => ({ ...next, kind: 'consumer' as const }))]) {
      const isImplicit = !consumers.includes(consumer as ProseUse);
      for (const producer of site.stages) {
        const consumerStages = stageCandidates(stages, (consumer as ProseUse).name!);
        for (const target of consumerStages) {
          // Several folders sharing a name (c/parse, python/parse): pair stages of the same folder.
          const same = (a: string, b: string) => a.slice(0, a.lastIndexOf('/')) === b.slice(0, b.lastIndexOf('/'));
          if ((site.stages.length > 1 || consumerStages.length > 1) && !same(producer.record.source, target.record.source)) continue;
          if (producer.record.source === target.record.source) continue;
          const id = `${scope}|${producer.record.source}|${target.record.source}|${site.start}`;
          if (seen.has(id)) continue;
          seen.add(id);
          const { param, paramType } = consumerParam(target, (consumer as ProseUse).position ?? 0);
          const others = uses.filter(use => use !== consumer);
          const readers: Reader[] = [{ kind: 'consumer', name: target.record.name, site: siteOf(scope, text, (consumer as ProseUse).at),
            detail: (consumer as ProseUse).detail, certain: !isImplicit },
            ...others.map(use => ({ kind: use.kind === 'consumer' ? 'other-call' as const : use.kind, ...(use.name ? { name: use.name } : {}),
              site: siteOf(scope, text, use.at), detail: use.detail, certain: !(use as { uncertain?: boolean }).uncertain })),
            ...(isImplicit ? [{ kind: 'eval' as const, site: siteOf(scope, text, site.start),
              detail: (consumer as { typed?: boolean }).typed ? 'only the declared types connect the calls; nothing shows who else reads the value' :
                'the chain is only implied by the prose; nothing shows who else reads the value', certain: false }] : [])];
          edges.push({ id, scope, scopeKind: 'nl', chain: `${producer.path} -> ${target.path}`, producer: refOf(producer.record),
            consumer: { ...refOf(target.record), param, paramType }, type: producer.record.returns,
            flow: isImplicit ? ((consumer as { typed?: boolean }).typed ? 'typed' : 'implicit') : flow, variable, dialect: DEFAULT_DIALECT, readers,
            producerSites: producerSites(producer), consumerSites: consumerCounts.get(target.record.source) ?? 0,
            alreadySoft: isSoft(producer.record.returns) || (paramType ? isSoft(paramType) : false),
            finite: isFinite_(producer.record.returns),
            excerpt: sentenceAround(text, site.start).slice(0, 400) });
        }
      }
    }
  }
  return edges;
}

/** "A(x), then B(...)" or "A(x) then B": the next stage call after a bare call, joined by a sequencing word. */
type ImplicitConsumer = { name: string; position: number; detail: string; at: number; typed?: boolean };

/** The first later call whose stage takes the producer's declared (non-primitive) result type as a parameter. */
function typedConsumers(_text: string, site: CallSite, sites: readonly CallSite[]): ImplicitConsumer[] {
  const types = new Set(site.stages.map(stage => stage.record.returns.replace(/\s+/g, ' ').trim()));
  if ([...types].some(type => /^(?:string|number|boolean|unknown|null|any)(?:\[\])?$/.test(type) || isSoft(type))) return [];
  for (const next of sites.filter(item => item.start > site.close).sort((a, b) => a.start - b.start)) {
    const position = Object.values(next.stages[0]!.record.args).findIndex(type => types.has(type.replace(/\s+/g, ' ').trim()));
    if (position >= 0) return [{ name: next.reference, position, typed: true, at: next.start,
      detail: `${next.reference} takes the ${[...types][0]} that ${site.reference} returns` }];
  }
  return [];
}

function implicitConsumers(text: string, site: CallSite, sites: readonly CallSite[]): ImplicitConsumer[] {
  const next = sites.filter(item => item.start > site.close).sort((a, b) => a.start - b.start)[0];
  if (!next) return [];
  const between = text.slice(site.close + 1, next.start);
  if (between.length > 120 || !/\bthen\b|\bnext\b|\bfollowed by\b|\bof the result\b|\bwith (?:that|the result)\b/i.test(between)) return [];
  if (/[.\n]\s*\n/.test(between)) return [];
  return [{ name: next.reference, position: 0, detail: `the next stage after ${site.reference}, joined by "${between.trim().slice(0, 40)}"`, at: next.start }];
}

// --- Code that calls named functions: crisp TypeScript orchestrators and recorded eval programs ------------------------

type Binding = { stage: NatlangRecord; path: string };

/**
 * What the analysis knows about the code it reads. `crisp-typescript`: an orchestrator written in TypeScript, where any
 * other function that receives a value is crisp code. `eval`: code the orchestrating model wrote and ran (recorded in the
 * call store), where a call to a host service is a `service` reader and any other use is the model reading the value.
 */
type CodeContext = { file: ts.SourceFile; fileName: string; imports: ReadonlyMap<string, ItemRecord>; mode: 'crisp-typescript' | 'eval';
  /** Line numbers are shifted by this many lines (a wrapper added above the code). */
  lineOffset: number };

function importedFunctions(file: ts.SourceFile, fileName: string, resolve: (from: string, specifier: string) => NatlangRecord | undefined):
    Map<string, NatlangRecord> {
  const out = new Map<string, NatlangRecord>();
  for (const statement of file.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
    const specifier = statement.moduleSpecifier.text;
    if (!specifier.endsWith('.nl')) continue;
    const record = resolve(fileName, specifier);
    const clause = statement.importClause;
    if (record && clause?.name) out.set(clause.name.text, record);
  }
  return out;
}

function memberStage(record: ItemRecord, names: string[]): NatlangRecord | undefined {
  let current: ItemRecord = record;
  for (const name of names) {
    const child: ItemRecord | undefined = current.codebase[name];
    if (!child) return undefined;
    current = child;
  }
  return current.kind === 'natlang' ? current : undefined;
}

/** The named function a call expression invokes: `plan(...)`, `database.plan(...)`. */
function calleeOf(call: ts.CallExpression, imports: ReadonlyMap<string, ItemRecord>): Binding | undefined {
  const names: string[] = [];
  let expression: ts.Expression = call.expression;
  while (ts.isPropertyAccessExpression(expression)) { names.unshift(expression.name.text); expression = expression.expression; }
  if (!ts.isIdentifier(expression)) return undefined;
  const root = imports.get(expression.text);
  if (!root) return undefined;
  const stage = names.length ? memberStage(root, names) : root.kind === 'natlang' ? root : undefined;
  return stage ? { stage, path: [expression.text, ...names].join('.') } : undefined;
}

const transparent = (node: ts.Node): boolean =>
  ts.isParenthesizedExpression(node) || ts.isAwaitExpression(node) || ts.isAsExpression(node) || ts.isNonNullExpression(node);
const unwrap = (node: ts.Node): ts.Node => {
  let current = node;
  while (transparent(current)) current = (current as ts.ParenthesizedExpression).expression;
  return current;
};

/** The call a value is an argument of, looking through await, parentheses and casts. */
function nestedCall(node: ts.Node): ts.CallExpression | undefined {
  for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
    if (ts.isCallExpression(current)) return current;
    if (!(ts.isParenthesizedExpression(current) || ts.isAwaitExpression(current) || ts.isAsExpression(current))) return undefined;
  }
  return undefined;
}

const argumentPosition = (call: ts.CallExpression, node: ts.Node): number =>
  call.arguments.findIndex(argument => argument === node || (node.pos >= argument.pos && node.end <= argument.end));

const lineIn = (ctx: CodeContext, node: ts.Node): number => ctx.file.getLineAndCharacterOfPosition(node.getStart(ctx.file)).line + 1 - ctx.lineOffset;
const siteIn = (ctx: CodeContext, node: ts.Node): string => `${ctx.fileName}:${lineIn(ctx, node)}`;

/** A use of a value in code: who reads it, and the function when it is one of the named functions. */
type CodeUse = { reader: Reader; consumer?: Binding; position?: number };

/** Who reads the value `node` (an identifier holding it, or the call that made it) in the place it is used. */
function useOf(ctx: CodeContext, node: ts.Node): CodeUse {
  const site = siteIn(ctx, node);
  const call = nestedCall(node);
  const asArgument = !!call && call.arguments.some(argument => unwrap(argument) === unwrap(node));
  const callee = call && asArgument ? calleeOf(call, ctx.imports) : undefined;
  if (callee) return { reader: { kind: 'consumer', name: callee.path, site, detail: `argument of ${callee.path}`, certain: true },
    consumer: callee, position: argumentPosition(call!, node) };
  if (ts.isReturnStatement(node.parent) || (node.parent && ts.isReturnStatement(node.parent.parent ?? node.parent)))
    return { reader: { kind: 'host-return', site, detail: ctx.mode === 'eval' ? 'returned by the orchestrator' : 'returned to the caller', certain: true } };
  if (call && asArgument) {
    const text = call.expression.getText(ctx.file);
    const crisp = ctx.mode === 'crisp-typescript' || ctx.imports.get(text.split('.')[0]!)?.kind === 'module';
    return { reader: { kind: crisp ? 'crisp-code' : 'service', name: text, site, detail: `passed to ${text}`, certain: true } };
  }
  return { reader: { kind: ctx.mode === 'eval' ? 'eval' : 'crisp-code', site,
    detail: `${ctx.mode === 'eval' ? 'read by the orchestrating model' : 'read by crisp code'}: ${node.parent.getText(ctx.file).slice(0, 60)}`, certain: true } };
}

/** How the result of one call of a named function travels in the code around it. */
type CallFlow = { form: 'nested' | 'bound' | 'loose'; variable: string | null; uses: CodeUse[]; anchor: ts.Node };

function flowOfCall(ctx: CodeContext, node: ts.CallExpression): CallFlow {
  // B(await A(x)): the value never has a name.
  const outer = nestedCall(node);
  const parent = node.parent;
  const direct = outer && outer !== node && (ts.isAwaitExpression(parent) || ts.isParenthesizedExpression(parent) || parent === outer);
  const consumer = direct ? calleeOf(outer!, ctx.imports) : undefined;
  if (consumer && outer!.arguments.some(argument => unwrap(argument) === node))
    return { form: 'nested', variable: null, anchor: node, uses: [{ reader: { kind: 'consumer', name: consumer.path, site: `${ctx.fileName}:${lineIn(ctx, outer!)}`,
      detail: `argument of ${consumer.path}`, certain: true }, consumer, position: argumentPosition(outer!, node) }] };
  // const v = await A(x): every reference to v in the enclosing function is a reader.
  let declaration: ts.Node | undefined = parent;
  while (declaration && transparent(declaration)) declaration = declaration.parent;
  if (declaration && ts.isVariableDeclaration(declaration) && ts.isIdentifier(declaration.name)) {
    const bound = declaration;
    const name = declaration.name.text;
    let region: ts.Node = ctx.file;
    for (let current: ts.Node | undefined = declaration.parent; current; current = current.parent)
      if (ts.isFunctionLike(current) || ts.isSourceFile(current)) { region = current; break; }
    const uses: CodeUse[] = [];
    const find = (item: ts.Node): void => {
      if (ts.isIdentifier(item) && item.text === name && item !== bound.name &&
          !(ts.isPropertyAccessExpression(item.parent) && item.parent.name === item)) uses.push(useOf(ctx, item));
      ts.forEachChild(item, find);
    };
    find(region);
    return { form: 'bound', variable: name, uses, anchor: declaration };
  }
  // Anything else (a statement of its own, a destructuring, part of a larger expression): the surrounding code reads it.
  return { form: 'loose', variable: null, uses: [useOf(ctx, node)], anchor: node };
}

function typescriptEdges(fileName: string, text: string, resolve: (from: string, specifier: string) => NatlangRecord | undefined): EdgeFacts[] {
  const file = ts.createSourceFile(fileName, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const imports = importedFunctions(file, fileName, resolve);
  if (!imports.size) return [];
  const ctx: CodeContext = { file, fileName, imports, mode: 'crisp-typescript', lineOffset: 0 };
  const edges: EdgeFacts[] = [];
  const functionName = (node: ts.Node): string => {
    for (let current: ts.Node | undefined = node.parent; current; current = current.parent) {
      if ((ts.isFunctionDeclaration(current) || ts.isMethodDeclaration(current)) && current.name) return current.name.getText(file);
      if (ts.isVariableDeclaration(current) && ts.isIdentifier(current.name) && current.initializer &&
        (ts.isArrowFunction(current.initializer) || ts.isFunctionExpression(current.initializer))) return current.name.text;
    }
    return '<module>';
  };
  const sitesIn = new Map<string, number>();
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const producer = calleeOf(node, imports);
      if (producer) sitesIn.set(producer.stage.source, (sitesIn.get(producer.stage.source) ?? 0) + 1);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  const addEdge = (producer: Binding, consumer: Binding, position: number, variable: string | null, flow: 'nested' | 'bound',
      readers: Reader[], anchor: ts.Node) => {
    const scope = `${fileName}#${functionName(anchor)}`;
    const { param, paramType } = consumerParam({ path: consumer.path, record: consumer.stage }, position);
    edges.push({ id: `${scope}|${producer.stage.source}|${consumer.stage.source}|${lineIn(ctx, anchor)}`, scope, scopeKind: 'typescript',
      chain: `${producer.path} -> ${consumer.path}`, producer: refOf(producer.stage), consumer: { ...refOf(consumer.stage), param, paramType },
      type: producer.stage.returns, flow, variable, dialect: DEFAULT_DIALECT, readers,
      producerSites: sitesIn.get(producer.stage.source) ?? 1, consumerSites: sitesIn.get(consumer.stage.source) ?? 1,
      alreadySoft: isSoft(producer.stage.returns) || (paramType ? isSoft(paramType) : false),
      finite: isFinite_(producer.stage.returns) });
  };
  const walk = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const producer = calleeOf(node, imports);
      if (producer) {
        const flow = flowOfCall(ctx, node);
        if (flow.form !== 'loose') for (const use of flow.uses) {
          if (use.reader.kind !== 'consumer' || !use.consumer) continue;
          const readers: Reader[] = flow.form === 'nested' ? [use.reader] :
            [use.reader, ...flow.uses.filter(other => other !== use).map(other => other.reader.kind === 'consumer' ? { ...other.reader, kind: 'other-call' as const } : other.reader)];
          addEdge(producer, use.consumer, use.position ?? 0, flow.variable, flow.form, readers, flow.anchor);
        }
      }
    }
    ts.forEachChild(node, walk);
  };
  walk(file);
  return edges;
}

// --- Observed readers ----------------------------------------------------------------------------------------------

/** One recorded run of an orchestrator: the eval programs the model ran, in order. */
export type ObservedRun = { id: string; evals: readonly string[] };

/**
 * Recorded runs of orchestrators, from the call store. `runs` returns the model-driven runs of the orchestrator at
 * `scope` that ran exactly these `instructions`, and a revision naming the store state they were read from (the plan
 * records it, so a later store can be told apart). `minRuns` is the number of supporting runs a plan will accept.
 */
export type ObservedSource = { minRuns: number;
  runs(scope: string, instructions: string): { revision: string; runs: readonly ObservedRun[] } };

/** What a plan is told about an edge whose readers were read from recorded eval code instead of proven from source. */
export type ObservedEvidence = {
  /** Runs in which every producer call fed only the consumer. */
  runs: number;
  /** Runs in which some producer call was read by something else (its readers are then listed on the edge). */
  contradicted: number;
  /** The number of supporting runs a plan accepts (the verifier's setting). */
  minRuns: number;
  /** The store state the runs were read from. */
  revision: string;
};

/** The call sites of the stages in one run's eval programs, with how each result travels. */
function evalSites(record: NatlangRecord, run: ObservedRun): { producer: Binding; flow: CallFlow; ctx: CodeContext }[] {
  const text = `export {};\nasync function __orchestrator() {\n${run.evals.join('\n;\n')}\n}\n`;
  const file = ts.createSourceFile(`eval:${run.id}`, text, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TS);
  const ctx: CodeContext = { file, fileName: `${record.source}@${run.id}`, imports: new Map(Object.entries(record.codebase)), mode: 'eval', lineOffset: 2 };
  const sites: { producer: Binding; flow: CallFlow; ctx: CodeContext }[] = [];
  const walk = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const producer = calleeOf(node, ctx.imports);
      if (producer) sites.push({ producer, flow: flowOfCall(ctx, node), ctx });
    }
    ts.forEachChild(node, walk);
  };
  walk(file);
  return sites;
}

/**
 * Replace the unknown reader of hand-offs the prose only implies (`typed` and `implicit` edges) by what the recorded runs
 * show. A run supports an edge when every call of the producer in it passes the value to the consumer's parameter and
 * to nothing else; it contradicts when some call of the producer is read by anything else (including the model's own
 * code, a service or the result of the orchestrator). The edge is proven by observation only when no run contradicts it;
 * the verifier then asks for enough supporting runs.
 */
function observeEdges(record: NatlangRecord, edges: EdgeFacts[], observed: ObservedSource): void {
  const candidates = edges.filter(edge => edge.scopeKind === 'nl' && (edge.flow === 'typed' || edge.flow === 'implicit') &&
    edge.readers.every(reader => reader.kind === 'consumer' || !reader.certain));
  if (!candidates.length) return;
  const { revision, runs } = observed.runs(record.source, record.instructions);
  if (!runs.length) return;
  const tallies = new Map<string, { supported: number; contradicted: number; outside: Reader[] }>(
    candidates.map(edge => [edge.id, { supported: 0, contradicted: 0, outside: [] }]));
  for (const run of runs) {
    const sites = evalSites(record, run);
    for (const edge of candidates) {
      const own = sites.filter(site => site.producer.stage.source === edge.producer.source);
      if (!own.length) continue;
      const tally = tallies.get(edge.id)!;
      const outside: Reader[] = [];
      for (const site of own) {
        const feeds = site.flow.uses.filter(use => use.consumer?.stage.source === edge.consumer.source &&
          consumerParam({ path: use.consumer.path, record: use.consumer.stage }, use.position ?? 0).param === edge.consumer.param);
        const others = site.flow.uses.filter(use => !feeds.includes(use));
        if (feeds.length !== 1) outside.push({ kind: 'eval', site: site.ctx.fileName,
          detail: feeds.length ? 'the model passes the value to the consumer more than once' : 'the model does not pass the value to the consumer', certain: true });
        for (const use of others) outside.push(use.reader.kind === 'consumer' ? { ...use.reader, kind: 'other-call' } : use.reader);
      }
      if (outside.length) { tally.contradicted++; tally.outside.push(...outside); } else tally.supported++;
    }
  }
  for (const edge of candidates) {
    const tally = tallies.get(edge.id)!;
    if (!tally.supported && !tally.contradicted) continue;
    edge.observed = { runs: tally.supported, contradicted: tally.contradicted, minRuns: observed.minRuns, revision };
    const consumer = edge.readers.find(reader => reader.kind === 'consumer')!;
    edge.readers = [{ ...consumer, certain: true, detail: `${consumer.detail}; seen in ${tally.supported} recorded run${tally.supported === 1 ? '' : 's'}` },
      ...tally.outside.slice(0, 5)];
  }
}

// --- The project -------------------------------------------------------------------------------------------------

/**
 * The candidate edges of the project at `root`. `files` reads sources (Node: `nodeSourceFiles(root)`).
 * `typescript` limits crisp-orchestrator analysis to the listed files when given (default: every `.ts` outside
 * test, vendor and build directories).
 */
export function fusionFacts(root: string, files: SourceFiles, options: { typescript?: readonly string[]; observed?: ObservedSource } = {}): FusionFacts {
  const skipped: FusionFacts['skipped'] = [];
  const nlFiles: string[] = [], tsFiles: string[] = [];
  const walk = (dir: string, insideFolder: boolean): void => {
    for (const entry of files.list(dir).sort()) {
      if (entry.startsWith('.') || SKIPPED.has(entry)) continue;
      const path = files.join(dir, entry);
      if (files.isDirectory(path)) { walk(path, insideFolder || files.isFile(`${path}.nl`)); continue; }
      if (entry.endsWith('.nl') && !insideFolder) nlFiles.push(path);
      else if (/\.tsx?$/.test(entry) && !entry.endsWith('.d.ts') && !entry.endsWith('.d.nl.ts')) tsFiles.push(path);
    }
  };
  walk(root, false);
  const records = new Map<string, NatlangRecord>();
  for (const path of nlFiles) {
    try { records.set(path, loadNamedFunction(path, files)); }
    catch (error) { skipped.push({ source: files.relative?.(path) ?? path, reason: error instanceof Error ? error.message.split('\n')[0]! : String(error) }); }
  }
  const edges: EdgeFacts[] = [], scopes: FusionFacts['scopes'] = [];
  const visitRecord = (record: NatlangRecord): void => {
    const stages = stagesOf(record);
    if (stages.length) {
      scopes.push({ scope: record.source, kind: 'nl', stages: stages.length });
      const found = proseEdges(record, stages);
      if (options.observed) observeEdges(record, found, options.observed);
      edges.push(...found);
    }
    for (const item of Object.values(record.codebase)) {
      if (item.kind === 'natlang') visitRecord(item);
    }
  };
  for (const record of records.values()) visitRecord(record);
  // Every named function, nested or not, by the file it came from.
  const byPath = new Map<string, NatlangRecord>();
  const index = (record: NatlangRecord): void => {
    byPath.set(files.join(root, record.source), record);
    for (const item of Object.values(record.codebase)) {
      if (item.kind === 'natlang') index(item);
      else for (const child of Object.values(item.codebase)) if (child.kind === 'natlang') index(child);
    }
  };
  for (const record of records.values()) index(record);
  const resolve = (from: string, specifier: string): NatlangRecord | undefined => {
    const path = files.join(files.dirname(from), specifier);
    return byPath.get(path) ?? byPath.get(path.replace(/\/\.\//g, '/'));
  };
  for (const path of options.typescript ?? tsFiles) {
    let text: string;
    try { text = files.read(path); } catch { continue; }
    if (!/\.nl['"]/.test(text)) continue;
    const found = typescriptEdges(files.relative?.(path) ?? path, text, (_from, specifier) => resolve(path, specifier));
    if (found.length) scopes.push({ scope: files.relative?.(path) ?? path, kind: 'typescript', stages: new Set(found.flatMap(edge => [edge.producer.source, edge.consumer.source])).size });
    edges.push(...found);
  }
  return { schema: FUSION_FACTS_SCHEMA, edges, scopes, skipped };
}
