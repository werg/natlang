/**
 * The current serving request of a recorded natlang tool-loop call: what this runtime sends for the same call. Training
 * data built from earlier recordings is rebuilt here, with the agent's own composition (callNotes, the tool list of
 * NativeToolAgent), so training rows and serving cannot drift apart. Python reaches it through
 * scripts/serving-request.mjs (training/neuralese/natlang_neuralese/data/serving_prompt.py).
 *
 * 1. Find every known prompt piece (this runtime's promptPieces() and the historical texts of each piece) in the
 *    recorded system prompt, longest first, without overlaps.
 * 2. The call is a tool-loop call iff an interpreter piece occurs. Its surface follows from the generic pieces found:
 *    depth limit, own code (function-tools), folder access and file-tool surface (directory-reducer/<surface>...).
 * 3. The system prompt becomes the entry point's base prompt (the same for every call). The recorded system text without
 *    its runtime-composed and teacher-only pieces (program declarations, guidance, other runtime notes) is the call's
 *    guidance; callNotes composes it with the surface's paragraphs as the last paragraph of the call's opening message.
 * 4. Tools become the surface's current tool list, return_result's value typed by the declared result of the call's
 *    signature in its opening (schemaOf, as the runtime does); when that cannot be derived, by the recording's own.
 *
 * A record that is not a tool-loop call, has no opening message, or calls a tool the current surface does not offer is
 * returned unchanged with the reason. Nothing here grants admission or changes split, source or target bytes.
 */
import { NativeToolAgent, callNotes, withCallNotes } from '../native/agent.js';
import { SYSTEM_PROMPT, type FileToolSurface } from '../native/prompt.js';
import { findPieces, promptPieces, renderPromptMessages, renderSystemPrompt, type PromptPiece, type PromptRendering } from '../native/system-prompts.js';
import { defaultSystemPrompt } from './collector.js';
import { hexDigest } from '../native/hash.js';
import { parseType, TypeEnv, type Type } from '../native/types.js';

/** The tool-loop entry points, `collector` (eval probes, RL and teacher collection) and `runtime` (natlang run); both
 * send SYSTEM_PROMPT since 2026-11-12 (natlang run sent TOOLS_PROMPT without GENERATION_GUIDANCE before). */
export const SERVING_ENTRY_POINTS = { collector: defaultSystemPrompt, runtime: SYSTEM_PROMPT } as const;
export type ServingEntry = keyof typeof SERVING_ENTRY_POINTS;

type Message = Record<string, unknown>;
type BankPiece = PromptPiece & { logical: string; historical: boolean };
export type CallSurface = { depthLimited: boolean; ownCode: boolean; folder: 'read' | 'write' | null; fileTools: FileToolSurface };

const INTERPRETERS = new Set(['interpreter', 'interpreter/depth-limit']);
/** Teacher-only appendices (teacher --approach-guide, generation guidance): the entry point's base supplies its own. */
const TEACHER_ONLY = new Set(['approach', 'generation-guidance']);
const composedPiece = (logical: string) => INTERPRETERS.has(logical) || logical === 'function-tools' ||
  (logical.startsWith('directory-reducer/') && logical !== 'directory-reducer/declaration');

/** `historical/<logical id>/<suffix>` -> the logical id (the longest current id it names). */
function logicalId(id: string, current: ReadonlySet<string>): string {
  if (!id.startsWith('historical/')) return id;
  const rest = id.slice('historical/'.length);
  return [...current].filter(key => rest.startsWith(key + '/')).sort((a, b) => b.length - a.length)[0] ?? rest.replace(/\/[^/]*$/, '');
}

/** This runtime's pieces plus earlier texts of them (training/prompt-pieces/pieces.json), each with its logical id. */
export function pieceBank(history: readonly PromptPiece[] = []): BankPiece[] {
  const current = promptPieces(), ids = new Set(current.map(piece => piece.id)), known = new Set<string>();
  const bank: BankPiece[] = [];
  for (const piece of current) { known.add(piece.text); bank.push({ ...piece, logical: piece.id, historical: false }); }
  for (const piece of history) {
    if (!piece.text || known.has(piece.text) || (piece.scope !== 'generic' && piece.scope !== 'program')) continue;
    known.add(piece.text);
    bank.push({ id: piece.id, text: piece.text, scope: piece.scope, logical: logicalId(piece.id, ids), historical: true });
  }
  return bank;
}

/** The tool-loop call surface named by the generic pieces of a recorded system prompt (undefined: not a tool loop). */
export function callSurface(found: readonly [number, number, BankPiece][]): CallSurface | undefined {
  const logical = found.filter(([, , piece]) => piece.scope === 'generic').map(([, , piece]) => piece.logical);
  const interpreter = logical.find(id => INTERPRETERS.has(id));
  if (!interpreter) return undefined;
  let depthLimited = interpreter === 'interpreter/depth-limit', folder: CallSurface['folder'] = null, fileTools = 'all';
  const reducer = logical.find(id => id.startsWith('directory-reducer/') && id !== 'directory-reducer/declaration');
  if (reducer) {
    const parts = reducer.split('/');
    fileTools = parts[1]!;
    folder = parts.includes('read-only') ? 'read' : 'write';
    depthLimited ||= parts.at(-1) === 'depth-limit';
  }
  return { depthLimited, ownCode: logical.includes('function-tools'), folder, fileTools: fileTools as FileToolSurface };
}

const CLOSE: Record<string, string> = { '(': ')', '[': ']', '{': '}', '<': '>' };

/** The declared result type text of `name(params): Result`. */
function returnText(signature: string): string {
  const open = signature.indexOf('(');
  if (open < 0) throw new Error('signature has no parameter list');
  const stack: string[] = [];
  for (let i = open; i < signature.length; i++) {
    const ch = signature[i]!;
    if (ch === '"' || ch === "'") {  // string literal types
      const end = signature.indexOf(ch, i + 1);
      if (end < 0) throw new Error('unterminated string literal in signature');
      i = end;
    } else if (CLOSE[ch]) stack.push(CLOSE[ch]!);
    else if (ch === stack.at(-1) && !(ch === '>' && signature[i - 1] === '=')) {
      stack.pop();
      if (!stack.length) {
        const rest = signature.slice(i + 1);
        if (!rest.startsWith(': ')) throw new Error('signature has no declared result');
        return rest.slice(2);
      }
    }
  }
  throw new Error('unbalanced signature');
}

/** The declared result type of a call opening ("You are inside this call: <signature>" and the `type` aliases under it). */
export function declaredResult(opening: string): { returns: Type; env: TypeEnv } {
  const lines = opening.split('\n'), head = 'You are inside this call: ';
  if (!lines[0]?.startsWith(head)) throw new Error('not a call opening');
  const names: Record<string, Type> = {};
  for (const line of lines.slice(1)) {
    if (!line.trim()) break;
    const alias = /^type ([A-Za-z_$][\w$]*) = ([\s\S]*);$/.exec(line);
    if (alias) names[alias[1]!] = parseType(alias[2]!);
  }
  return { returns: parseType(returnText(lines[0].slice(head.length))), env: new TypeEnv(names) };
}

/** The tool list this runtime offers a call of `surface` with declared result `result` (NativeToolAgent.tools). */
export function servingTools(surface: CallSurface, result?: { returns: Type; env: TypeEnv }): Record<string, unknown>[] {
  const session = { rememberOfferedTools() {}, runtime: { frame: { adHocDepth: surface.depthLimited ? 1e9 : 0 } },
    env: result?.env ?? new TypeEnv(),
    lam: { type: result ? { kind: 'lambda', params: { kind: 'object', fields: [] }, returns: result.returns } : { kind: 'opaque' },
      codebase: surface.ownCode ? { f: {} } : {},
      projectTransaction: surface.folder ? { folder: { access: surface.folder } } : undefined } };
  return new NativeToolAgent(async () => ({ calls: [] }), { fileTools: surface.fileTools })
    .tools(session as unknown as Parameters<NativeToolAgent['tools']>[0]) as unknown as Record<string, unknown>[];
}

const calledTools = (messages: readonly unknown[]) => new Set(messages.flatMap(message =>
  ((message as Message | undefined)?.tool_calls as Message[] | undefined ?? []).map(call =>
    ((call.function ?? call) as Message).name).filter((name): name is string => typeof name === 'string')));

export type ServingReport = Record<string, unknown> & { mode: string; reason?: string };

/** The current serving request of `entry` for a recorded call (module comment). */
export function servingRequest(recorded: { messages: Message[]; tools?: unknown[]; target?: Message },
  options: { entry: ServingEntry; bank: readonly BankPiece[]; rendering?: PromptRendering }):
  { messages: Message[]; tools: unknown[]; report: ServingReport } {
  const { messages, tools = [], target } = recorded;
  const base = SERVING_ENTRY_POINTS[options.entry];
  if (base === undefined) throw new Error(`unknown serving entry point ${JSON.stringify(options.entry)}`);
  const report = { entry: options.entry };
  const unchanged = (reason: string, extra: Record<string, unknown> = {}) =>
    ({ messages, tools, report: { ...report, mode: 'recorded', reason, ...extra } });
  const system = messages[0];
  if (system?.role !== 'system' || typeof system.content !== 'string') return unchanged('no plain-text system turn');
  const text = system.content;
  const found = findPieces(text, options.bank);
  const surface = callSurface(found);
  if (!surface) return unchanged('not a tool-loop call');
  const opening = messages.findIndex(message => message.role === 'user');
  if (opening < 0 || typeof messages[opening]!.content !== 'string') return unchanged('no plain-text call opening');
  // A recording of this layout (its call notes already in the opening) is what the runtime sends; it is not rebuilt.
  if (findPieces(messages[opening]!.content as string, options.bank)
    .some(([, , piece]) => piece.scope === 'generic' && composedPiece(piece.logical))) return unchanged('notes already in the call opening');
  let offered: Record<string, unknown>[], returnSchema = 'derived';
  try { offered = servingTools(surface, declaredResult(messages[opening]!.content as string)); } catch (error) {
    // Not derivable from the opening: the recording's own value schema of return_result, if it has one.
    const recordedValue = (tools as Message[]).map(tool => (tool?.function ?? tool) as Message)
      .find(fn => fn?.name === 'return_result')?.parameters as Message | undefined;
    const value = (recordedValue?.properties as Message | undefined)?.value as Message | undefined;
    if (!value || typeof value !== 'object') return unchanged('return type not derivable', { detail: String((error as Error).message).slice(0, 200) });
    offered = servingTools(surface);
    const last = offered.at(-1)!.function as { parameters: { properties: Record<string, Message> } };
    const { description: _description, ...typed } = value;
    last.parameters.properties.value = { ...typed, description: last.parameters.properties.value!.description };
    returnSchema = 'recorded';
  }
  const names = new Set(offered.map(tool => ((tool.function ?? tool) as Message).name));
  const missing = [...calledTools([...messages, ...(target ? [target] : [])])].filter(name => !names.has(name)).sort();
  if (missing.length) return unchanged('calls tools the surface lacks', { tools: missing });
  let guidance = '', cursor = 0;
  const removed: string[] = [], legacy = new Set<string>();
  const current = new Set(promptPieces().map(piece => piece.id));
  for (const [start, end, piece] of found) {
    if (piece.scope !== 'generic' || !(composedPiece(piece.logical) || TEACHER_ONLY.has(piece.logical))) {
      if (piece.historical && !current.has(piece.logical)) legacy.add(piece.logical);
      continue;  // program declarations and other runtime notes stay verbatim
    }
    guidance += text.slice(cursor, start) + '\n\n';
    removed.push(piece.logical);
    cursor = end;
  }
  guidance = (guidance + text.slice(cursor)).replace(/\n{3,}/g, '\n\n').trim();
  const notes = callNotes({ guidance, ownCode: surface.ownCode, allowAdHoc: !surface.depthLimited, rendering: options.rendering,
    folder: surface.folder ? { fileTools: surface.fileTools, writable: surface.folder === 'write' } : undefined });
  const user = messages[opening]!;
  const rebuilt = [{ ...system, content: base }, ...messages.slice(1, opening),
    { ...user, content: withCallNotes(user.content as string, notes) }, ...messages.slice(opening + 1)];
  const rendering = options.rendering ?? 'full';
  return { messages: renderPromptMessages(rebuilt, rendering), tools: offered, report: { ...report,
    mode: rendering === 'full' ? `serving-${options.entry}` : rendering, call_surface: surface, return_schema: returnSchema, removed,
    legacy_program_pieces: [...legacy].sort(), recorded_system_sha256: hexDigest(text),
    serving_system_sha256: hexDigest(base), opening_notes_sha256: hexDigest(notes) } };
}

/** renderSystemPrompt over this runtime's pieces and their earlier texts. */
export const renderRecordedSystemPrompt = (text: string, mode: PromptRendering, bank: readonly BankPiece[]) =>
  renderSystemPrompt(text, mode, [...bank]);
