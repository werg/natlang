/**
 * The companion's documents and its view of the workspace, shared by its background task (index.ts), its helpers on
 * the agent's live stream (stream.ts) and its help beside the user's draft (draft.ts).
 */
import type { Context } from '@earendil-works/chord';
import { defineDoc } from '../../vendor/durable/src/documents.ts';
import type { DocumentReader } from '../../vendor/durable/src/types.ts';
import type { ExecutionEnv } from '../../vendor/durable/src/env/index.ts';
import { isAbsolutePath, relativePath, resolvePath } from '../../host/paths.ts';
import type { Message, ToolCall } from '@earendil-works/pi-ai';
import { transcriptText } from '../../host/natlang-provider.ts';
import type { Briefing, FileKnowledge, FileSummary } from '../../types.ts';

/**
 * The briefing shown to this conversation's agent, and the entry it was written after; `research`: what the companion
 * looked up while the agent's last reply streamed (stream.ts), for the turn (generation task) that reply ended, as
 * notes the next request shows.
 */
export const CompanionDoc = defineDoc<{ briefing?: Briefing; basis?: number; research?: { turn: number; notes: string[] } }>({
  kind: 'pi.companion', version: 1, scope: 'conversation', history: 'latest', fork: 'current', initial: () => ({}),
});

/** What the companion knows about workspace files, shared by every conversation of the session. */
export const CompanionFiles = defineDoc<{ files: Record<string, FileKnowledge> }>({
  kind: 'pi.companion.files', version: 1, scope: 'session', initial: () => ({ files: {} }),
});

/** Lines a search returns. */
export const SEARCH_LINES = 40;
/** Characters of grep's output a search reads. */
const SEARCH_BYTES = 4 << 20;
/** Characters of a file the companion reads. */
export const FILE_LIMIT = 24_000;

const READ_DECLARATION = `/** The workspace as the companion sees it. Paths are relative to the workspace. */
/** A file's current hash and text (cut at ${FILE_LIMIT} characters), and what you know about this version (null when you know nothing about it, or only about an older version). Null when the file does not exist. */
export function file(path: string): Promise<{ hash: string; text: string; known: FileKnowledge | null } | null>;`;
const REMEMBER_DECLARATION = `
/** Remember summary as what you know about path, as file(path) last showed it. */
export function remember(path: string, summary: FileSummary): Promise<void>;`;
const SEARCH_DECLARATION = `
/** Lines matching the regular expression pattern (grep -E syntax) in the workspace's text files, as "path:line:text", at most ${SEARCH_LINES}; glob limits the files (for example "*.py"). Hidden directories are skipped. */
export function search(pattern: string, glob?: string): Promise<string[]>;
/** The entries of a workspace directory ("." for the root), directories with a trailing "/". Hidden entries are skipped. */
export function list(directory: string): Promise<string[]>;`;

/** The companion service as its background task declares it: reads, searches and remembering. */
export const COMPANION_DECLARATION = READ_DECLARATION + REMEMBER_DECLARATION + SEARCH_DECLARATION;
/** The read-only companion service (help beside a draft): no remembering. */
export const COMPANION_READ_DECLARATION = READ_DECLARATION + SEARCH_DECLARATION;

/** A grep -E pattern for a definition of `symbol` (an identifier): a definition keyword, then the name. */
export function definitionPattern(symbol: string): string {
  return `(function|class|interface|type|enum|const|let|var|def|struct|fn|func)[[:space:]]+${symbol.replace(/\$/g, '\\$')}([^[:alnum:]_$]|$)`;
}

/** A file version's identity: the first 16 hex digits of its text's SHA-256. */
export async function hashOf(text: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)));
  return Array.from(digest.subarray(0, 8), byte => byte.toString(16).padStart(2, '0')).join('');
}

/** The value of an environment result, or its error thrown. */
export function valueOf<T>(result: { ok: true; value: T } | { ok: false; error: unknown }): T {
  if (!result.ok) throw result.error;
  return result.value;
}

/** `path` relative to the workspace `cwd`, or undefined when it leaves the workspace. */
export function workspacePath(cwd: string, path: string): string | undefined {
  const local = relativePath(cwd, resolvePath(cwd, path));
  return !local || local.startsWith('..') || isAbsolutePath(local) ? undefined : local;
}

export type FileView = { hash: string; text: string; known: FileKnowledge | null };

/**
 * The companion's view of the workspace: the conversation's execution environment (`env`, rooted at `cwd`), so it reads
 * the files the agent's tools change and searches them with the environment's own grep, on any host. Without an
 * environment (the agent's tools then have none either) no file exists and searches find nothing. `read` gives what the
 * companion already knows (`CompanionFiles`); `remember` keeps a summary of the version `file` last showed (the
 * background task commits it, a speculative helper holds it until the agent's message confirms it). Without `remember`
 * the service is read-only.
 */
export function companionService(read: DocumentReader, context: Context, env: ExecutionEnv | undefined, cwd: string,
    remember?: (knowledge: FileKnowledge) => Promise<void>) {
  // The version of each file this run showed the model: what it summarizes is that version.
  const shown = new Map<string, string>();
  /** `path` (relative to the workspace) as an absolute path inside it, or undefined when it leaves the workspace. */
  const inside = (path: string) => {
    const absolute = resolvePath(cwd, path), local = relativePath(cwd, absolute);
    return local.startsWith('..') || isAbsolutePath(local) ? undefined : absolute;
  };
  const service = {
    async file(path: string): Promise<FileView | null> {
      if (!env) return null;
      const absolute = resolvePath(cwd, path);
      const info = await env.fileInfo(absolute, context);
      if (!info.ok || info.value.kind !== 'file') return null;
      const full = valueOf(await env.readTextFile(absolute, context));
      const hash = await hashOf(full);
      shown.set(path, hash);
      const known = (await read.snapshot(CompanionFiles, context))?.files[path];
      const text = full.length > FILE_LIMIT ? `${full.slice(0, FILE_LIMIT)}\n[cut: ${full.length - FILE_LIMIT} more characters]` : full;
      return { hash, text, known: known?.hash === hash ? known : null };
    },
    async search(pattern: string, glob?: string): Promise<string[]> {
      // grep itself, run by the environment without a shell. What every grep the environments run understands: no -I
      // (a binary match is reported on stderr, which is not read), and hidden directories as `.?*`, which leaves out
      // the searched directory `.` itself.
      if (!env) return [];
      const args = ['grep', '-rnE', '--exclude-dir=.?*', '-m', '5', ...(glob ? [`--include=${glob}`] : []), '-e', pattern, '.'];
      let output = '';
      const ran = await env.exec(args, { cwd, timeout: 10, onOutput: (text, _context, info) => {
        if (info.stream === 'stdout' && output.length < SEARCH_BYTES) output += text;
      } }, context);
      if (!ran.ok && ran.error.code !== 'timeout') return [];
      return output.split('\n').filter(Boolean).slice(0, SEARCH_LINES).map(line => line.replace(/^\.\//, '').slice(0, 300));
    },
    async list(directory: string): Promise<string[]> {
      const absolute = inside(directory);
      if (absolute === undefined) throw new Error(`${directory} is outside the workspace`);
      if (!env) throw new Error('No execution environment is configured');
      return valueOf(await env.listDir(absolute, context)).filter(entry => !entry.name.startsWith('.'))
        .map(entry => entry.kind === 'directory' ? `${entry.name}/` : entry.name).sort().slice(0, 200);
    },
  };
  if (!remember) return service;
  return {
    ...service,
    async remember(path: string, summary: FileSummary): Promise<void> {
      const hash = shown.get(path);
      if (hash === undefined) throw new Error(`remember(${JSON.stringify(path)}) needs file(${JSON.stringify(path)}) first: it remembers the version file showed you`);
      await remember(JSON.parse(JSON.stringify({ path, hash, purpose: summary.purpose, symbols: summary.symbols, notes: summary.notes })) as FileKnowledge);
    },
  };
}

/** A message part as the transcript reads it; Neuralese blocks (types.ts NeuraleseContent) are not in pi-ai's union. */
type Part = { type: string; text?: string; id?: string };

/** The text of a message, compact: tool calls with their arguments, results cut; a Neuralese block named by its ID. */
export function renderMessage(message: Message): string {
  const cut = (text: string, limit: number) => text.length > limit ? `${text.slice(0, limit)} …[${text.length - limit} more chars]` : text;
  if (message.role === 'user') return `USER: ${cut(typeof message.content === 'string' ? message.content : (message.content as Part[]).map(transcriptText).join(''), 2000)}`;
  if (message.role === 'assistant') return (message.content as Part[]).map(part => part.type === 'text' || part.type === 'neuralese' ?
    `AGENT: ${cut(transcriptText(part), 1500)}` : part.type === 'toolCall' ? `AGENT CALLS ${(part as ToolCall).name} ` +
    `${cut(JSON.stringify((part as ToolCall).arguments), 400)}` : '').filter(Boolean).join('\n');
  if (message.role === 'toolResult') return `RESULT of ${message.toolName}${message.isError ? ' (error)' : ''}: ${cut((message.content as Part[]).map(transcriptText).join(''), 800)}`;
  return '';
}

