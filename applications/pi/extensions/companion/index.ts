/**
 * The companion (COMPANION.md): a second mind beside the agent. After each tool round it runs in the background as a
 * durable `pi.companion` task of the conversation: it learns the workspace files the agent touched and writes a
 * briefing, which the `companion` section shows in the agent's next request. The companion's judgment is natural
 * language (`observe.nl`, `observe/summarize.nl`); this file is the mechanism: documents, the task, the trigger and
 * the section. Without this extension a conversation is today's harness.
 */
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { isAbsolute, join, relative, resolve } from 'node:path';
import type { Context } from '@earendil-works/chord';
import type { Message } from '@earendil-works/pi-ai';
import type { NatlangRuntime } from '@natlang/node';
import { defineDoc, defineDocFamily } from '../../vendor/durable/src/documents.ts';
import { GenerationTask } from '../../vendor/durable/src/harness/generation.ts';
import { ToolTask } from '../../vendor/durable/src/harness/tool.ts';
import { hook } from '../../vendor/durable/src/harness/define.ts';
import type { Harness } from '../../vendor/durable/src/harness/harness.ts';
import type { Extension, ToolExecutionResult } from '../../vendor/durable/src/harness/types.ts';
import { defineTask } from '../../vendor/durable/src/tasks.ts';
import type { ConversationId, TaskRuntime } from '../../vendor/durable/src/types.ts';
import type { Briefing, FileKnowledge, FileSummary, Observation } from '../../types.ts';
import observe from './observe.nl';

/** The briefing shown to this conversation's agent, and the entry it was written after. */
export const CompanionDoc = defineDoc<{ briefing?: Briefing; basis?: number }>({
  kind: 'pi.companion', version: 1, scope: 'conversation', history: 'latest', fork: 'current', initial: () => ({}),
});

/** What the companion knows about workspace files, shared by every conversation of the session. */
export const CompanionFiles = defineDoc<{ files: Record<string, FileKnowledge> }>({
  kind: 'pi.companion.files', version: 1, scope: 'session', initial: () => ({ files: {} }),
});

/** The full text of a tool output the agent saw shaped, by tool call ID (its recall handle). */
export const CompanionOutputs = defineDocFamily<{ tool: string; text: string }, { tool: string; text: string }>({
  kind: 'pi.companion.output', version: 1, scope: 'conversation', history: 'latest', fork: 'current', family: true,
  initial: seed => seed,
});

const TASK = 'pi.companion';
/** Tool outputs longer than this many characters reach the agent shaped: their head and tail, and a recall handle. */
const SHAPE_LIMIT = 6_000;
const SHAPE_HEAD = 2_500;
const SHAPE_TAIL = 2_000;
/** Messages of the transcript tail an observation shows. */
const RECENT_MESSAGES = 16;
/** Characters of a file the companion reads. */
const FILE_LIMIT = 24_000;

type Runtime = TaskRuntime<{ basis?: number }, { phase: 'observe' }, null, object>;

export const COMPANION_DECLARATION = `/** The workspace as the companion sees it. Paths are relative to the workspace. */
/** A file's current hash and text (cut at ${FILE_LIMIT} characters), and what you know about this version (null when you know nothing about it, or only about an older version). Null when the file does not exist. */
export function file(path: string): Promise<{ hash: string; text: string; known: FileKnowledge | null } | null>;
/** Remember summary as what you know about path, as file(path) last showed it. */
export function remember(path: string, summary: FileSummary): Promise<void>;`;

const hashOf = (text: string) => createHash('sha256').update(text).digest('hex').slice(0, 16);

/** The text of a message, compact: tool calls with their arguments, results cut. */
function render(message: Message): string {
  const cut = (text: string, limit: number) => text.length > limit ? `${text.slice(0, limit)} …[${text.length - limit} more chars]` : text;
  if (message.role === 'user') return `USER: ${cut(typeof message.content === 'string' ? message.content : message.content.map(part => part.type === 'text' ? part.text : '[image]').join(''), 2000)}`;
  if (message.role === 'assistant') return message.content.map(part => part.type === 'text' ? `AGENT: ${cut(part.text, 1500)}` :
    part.type === 'toolCall' ? `AGENT CALLS ${part.name} ${cut(JSON.stringify(part.arguments), 400)}` : '').filter(Boolean).join('\n');
  if (message.role === 'toolResult') return `RESULT of ${message.toolName}${message.isError ? ' (error)' : ''}: ${cut(message.content.map(part => part.type === 'text' ? part.text : '[image]').join(''), 800)}`;
  return '';
}

/** Workspace paths the tool calls of `messages` read, edited or wrote. */
function touched(messages: readonly Message[], cwd: string): string[] {
  const paths: string[] = [];
  for (const message of messages) {
    if (message.role !== 'assistant') continue;
    for (const part of message.content) {
      if (part.type !== 'toolCall' || !['read', 'edit', 'write'].includes(part.name)) continue;
      const path = (part.arguments as { path?: unknown }).path;
      if (typeof path !== 'string' || !path) continue;
      const local = relative(cwd, resolve(cwd, path));
      if (local && !local.startsWith('..') && !isAbsolute(local) && !paths.includes(local)) paths.push(local);
    }
  }
  return paths;
}

function companionService(runtime: Runtime, context: Context, cwd: string) {
  // The version of each file this run showed the model: what it summarizes is that version.
  const shown = new Map<string, string>();
  return {
    async file(path: string): Promise<{ hash: string; text: string; known: FileKnowledge | null } | null> {
      const absolute = join(cwd, path);
      try { if (!statSync(absolute).isFile()) return null; } catch { return null; }
      const full = readFileSync(absolute, 'utf8');
      const hash = hashOf(full);
      shown.set(path, hash);
      const known = (await runtime.snapshot(CompanionFiles, context))?.files[path];
      const text = full.length > FILE_LIMIT ? `${full.slice(0, FILE_LIMIT)}\n[cut: ${full.length - FILE_LIMIT} more characters]` : full;
      return { hash, text, known: known?.hash === hash ? known : null };
    },
    async remember(path: string, summary: FileSummary): Promise<void> {
      const hash = shown.get(path);
      if (hash === undefined) throw new Error(`remember(${JSON.stringify(path)}) needs file(${JSON.stringify(path)}) first: it remembers the version file showed you`);
      const knowledge = JSON.parse(JSON.stringify({ path, hash, purpose: summary.purpose, symbols: summary.symbols, notes: summary.notes })) as FileKnowledge;
      await runtime.commit(async tx => { (await tx.doc(CompanionFiles)).files[path] = knowledge as never; return undefined; }, context);
    },
  };
}

/**
 * A long output as the agent sees it: its head and tail around a note naming the recall handle. Which part to keep is
 * mechanism here (head and tail, like pi's own truncation); what the elided part means is the briefing's job.
 */
export function shapeOutput(text: string, handle: string): string {
  const elided = text.length - SHAPE_HEAD - SHAPE_TAIL;
  return `${text.slice(0, SHAPE_HEAD)}\n[… ${elided} characters elided by the companion; recall("${handle}") returns the full output …]\n${text.slice(-SHAPE_TAIL)}`;
}

/** The briefing as the agent reads it. Deterministic: an unchanged briefing renders the same text. */
export function briefingText(briefing: Briefing): string {
  // Stored documents drop empty lists, so a list may be absent.
  const list = (title: string, items: string[] | undefined) => items?.length ? [`${title}:`, ...items.map(item => `- ${item}`)] : [];
  return ['Notes from your harness companion, which watches your work in the background. Use what helps; ignore the rest.',
    `Focus: ${briefing.focus}`, ...list('Known', briefing.facts), ...list('Watch out', briefing.warnings),
    ...list('Consider', briefing.suggestions)].join('\n');
}

export type CompanionOptions = {
  /** The opened Harness, for starting companion tasks from a hook (hooks have no commit). */
  harness(): Harness;
  /** Called with each failure the companion keeps going after. */
  onReport?(error: unknown): void;
};

/** The companion extension; its functions run on `natlang`. */
export function companion(natlang: NatlangRuntime, options: CompanionOptions): Extension {
  const task = defineTask<{ basis?: number }, { phase: 'observe' }, null, object>({
    name: TASK, version: 1, initial: () => ({ phase: 'observe' }),
    phases: {
      async observe(current, runtime, context) {
        const agent = await runtime.agent(context);
        const cwd = agent.cwd ?? process.cwd();
        const view = await runtime.context(runtime.conversationId, context);
        const messages = view.messages as Message[];
        const goal = [...messages].reverse().find(message => message.role === 'user');
        const recent = messages.slice(-RECENT_MESSAGES);
        const observation: Observation = JSON.parse(JSON.stringify({
          goal: goal ? render(goal).replace(/^USER: /, '') : '',
          recent: recent.map(render).filter(Boolean).join('\n'),
          touched: touched(recent, cwd),
          previous: (await runtime.snapshot(CompanionDoc, runtime.conversationId, context))?.briefing ?? null,
          known: Object.values((await runtime.snapshot(CompanionFiles, context))?.files ?? {}).slice(-40),
        }));
        let briefing: Briefing | undefined, failure: string | undefined;
        try {
          briefing = await natlang.run(() => observe(observation as never), {
            services: { companion: companionService(runtime as Runtime, context, cwd) },
            serviceDeclarations: { companion: COMPANION_DECLARATION }, signal: runtime.signal,
            name: `${TASK}#${current.id}` }) as Briefing;
        } catch (error) {
          if (runtime.signal.aborted) throw error;
          failure = error instanceof Error ? error.message : String(error);
          options.onReport?.(error);
        }
        await runtime.commit(async tx => {
          if (briefing) {
            const doc = await tx.doc(CompanionDoc, runtime.conversationId);
            doc.briefing = JSON.parse(JSON.stringify(briefing)) as never;
            if (current.input.basis !== undefined) doc.basis = current.input.basis;
            return { status: 'terminal', outcome: { status: 'completed', result: null } };
          }
          return { status: 'terminal', outcome: { status: 'failed', error: { message: failure ?? 'the companion returned nothing' } } };
        }, context);
      },
    },
    async abort(_current, runtime, context) {
      await runtime.commit(() => ({ status: 'terminal', outcome: { status: 'aborted' } }), context);
    },
  });

  /** Start a companion run for the conversation unless one is pending or running. */
  const start = async (conversationId: ConversationId, basis: number | undefined, context: Context) => {
    try {
      const conversation = await options.harness().conversation(conversationId, context);
      await conversation?.commit(async tx => {
        for (const status of ['pending', 'running'] as const)
          if ((await tx.scanTasks({ conversationId, kind: TASK, status }, 1)).items.length) return;
        await tx.createTask(task, basis === undefined ? {} : { basis }, { ownership: { kind: 'conversation' }, conversationId, background: true });
      }, context);
    } catch (error) { options.onReport?.(error); }
  };

  return {
    name: 'companion',
    tasks: [task],
    tools: [{
      name: 'recall',
      description: 'Return the full text of a tool output that was shown shortened, by the handle its note names.',
      parameters: { type: 'object', properties: { handle: { type: 'string', description: 'The handle from the note in the shortened output' } }, required: ['handle'] } as never,
      replay: 'safe',
      async execute(args, api, context): Promise<ToolExecutionResult> {
        const handle = String((args as { handle?: unknown }).handle ?? '');
        const stored = await options.harness().snapshot(CompanionOutputs, api.conversationId, handle, context);
        if (!stored) return { content: [{ type: 'text', text: `No shortened output has the handle ${JSON.stringify(handle)}.` }], isError: true };
        return { content: [{ type: 'text', text: stored.text }] };
      },
    }],
    sections: [{
      key: 'companion',
      async render(input, context) {
        const briefing = (await input.read.snapshot(CompanionDoc, input.conversationId, context))?.briefing;
        return briefing ? briefingText(briefing) : undefined;
      },
    }],
    hooks: [hook(ToolTask, {
      // A long output is kept whole in the companion's store and reaches the agent shaped (COMPANION.md §1.3).
      async afterTool(call, result, api, context) {
        const parts = result.content ?? [];
        const text = parts.map(part => part.type === 'text' ? part.text : '').join('');
        if (call.name === 'recall' || text.length <= SHAPE_LIMIT) return undefined;
        try {
          const conversation = await options.harness().conversation(api.conversationId, context);
          await conversation?.commit(async tx => { await tx.doc(CompanionOutputs, api.conversationId, call.id, { tool: call.name, text }); }, context);
        } catch (error) { options.onReport?.(error); return undefined; }
        return { ...result, content: [{ type: 'text', text: shapeOutput(text, call.id) }, ...parts.filter(part => part.type !== 'text')] };
      },
    }), hook(GenerationTask, {
      // After every tool round: the agent has new information, and the companion catches up while it thinks.
      async afterTools(assistant, _results, api, context) { await start(api.conversationId, assistant as number, context); },
    })],
  };
}
