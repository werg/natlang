/**
 * The companion (COMPANION.md): a second mind beside the agent. After each tool round it runs in the background as a
 * durable `pi.companion` task of the conversation: it learns the workspace files the agent touched and writes a
 * briefing, which the `companion` section shows in the agent's next request. The companion's judgment is natural
 * language (`observe.nl`, `summarize.nl`); this file is the mechanism: documents, the task, the trigger and
 * the section. Without this extension a conversation is today's harness.
 */
import type { Context } from '@earendil-works/chord';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { AssistantMessage, Message, Models, ToolCall } from '@earendil-works/pi-ai';
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableSetting } from 'natlang:runtime';
import { GenerationTask } from '../../vendor/durable/src/harness/generation.ts';
import { ToolTask } from '../../vendor/durable/src/harness/tool.ts';
import { hook } from '../../vendor/durable/src/harness/define.ts';
import type { Harness } from '../../vendor/durable/src/harness/harness.ts';
import type { Extension, HookApi, ToolExecutionResult } from '../../vendor/durable/src/harness/types.ts';
import { defineTask } from '../../vendor/durable/src/tasks.ts';
import type { ConversationId } from '../../vendor/durable/src/types.ts';
import { ProviderDoc } from '../../vendor/durable/src/harness/provider.ts';
import type { Briefing, Observation, OutputShape } from '../../types.ts';
import { modelReader } from '../../host/natlang-provider.ts';
import { forceView, ToolOutputs, VIEW_LIMIT, ViewIntents, viewIntent, type ViewDocs } from '../../host/views.ts';
import { COMPANION_DECLARATION, CompanionDoc, CompanionFiles, companionService, renderMessage, workspacePath } from './workspace.ts';
import { CompanionStream, hintsOf, type SpeculationEvent } from './stream.ts';
import observe from './observe.nl';
import shape from './shape.nl';

/**
 * The full text of a long tool output, by tool call ID (its recall handle): the value input of the output's stored view
 * call (host/views.ts).
 */
export const CompanionOutputs = ToolOutputs;

export { COMPANION_DECLARATION, CompanionDoc, CompanionFiles, hintsOf, type SpeculationEvent };

const TASK = 'pi.companion';
/**
 * Tool outputs longer than this many characters reach a text reader shaped: their head and tail, and a recall handle.
 * Outputs longer than VIEW_LIMIT (host/views.ts) are stored as view calls, which a Neuralese reader reads as blocks.
 */
const SHAPE_LIMIT = 6_000;
const SHAPE_HEAD = 2_500;
const SHAPE_TAIL = 2_000;
/** Messages of the transcript tail an observation shows. */
const RECENT_MESSAGES = 16;

/** Workspace paths the tool calls of `messages` read, edited or wrote. */
function touched(messages: readonly Message[], cwd: string): string[] {
  const paths: string[] = [];
  for (const message of messages) {
    if (message.role !== 'assistant') continue;
    for (const part of message.content) {
      if (part.type !== 'toolCall' || !['read', 'edit', 'write'].includes(part.name)) continue;
      const path = (part.arguments as { path?: unknown }).path;
      if (typeof path !== 'string' || !path) continue;
      const local = workspacePath(cwd, path);
      if (local && !paths.includes(local)) paths.push(local);
    }
  }
  return paths;
}

/**
 * A long output as the agent sees it, crisp: its head and tail around a note naming the recall handle (like pi's own
 * truncation). The natural-language twin, `shape.nl`, chooses the lines; `renderShape` shows its choice.
 */
export function shapeOutput(text: string, handle: string): string {
  const elided = text.length - SHAPE_HEAD - SHAPE_TAIL;
  return `${text.slice(0, SHAPE_HEAD)}\n[… ${elided} characters elided by the companion; recall("${handle}") returns the full output …]\n${text.slice(-SHAPE_TAIL)}`;
}

/**
 * A long output as the agent sees it, by `shape`: the gist and recall note, then the kept lines in order with a mark
 * for each gap. Mechanism: ranges are clamped and merged, and the kept lines stop at `budget` characters whatever the
 * shape asked for.
 */
export function renderShape(text: string, handle: string, shape: OutputShape, budget = SHAPE_HEAD + SHAPE_TAIL): string {
  const lines = text.split('\n');
  const ranges = (shape.keep ?? []).map(range => [Math.max(1, Math.floor(range.from)), Math.min(lines.length, Math.floor(range.to))] as const)
    .filter(([from, to]) => from <= to).sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const [from, to] of ranges) {
    const last = merged.at(-1);
    if (last && from <= last[1] + 1) last[1] = Math.max(last[1], to); else merged.push([from, to]);
  }
  const out = [`[The companion shortened this output of ${lines.length} lines: ${shape.gist.trim()} recall("${handle}") returns it whole.]`];
  let used = 0, next = 1;
  for (const [from, to] of merged) {
    let line = from;
    for (; line <= to && used + lines[line - 1]!.length + 1 <= budget; line++) {
      if (line === from && from > next) out.push(`[… lines ${next}–${from - 1} …]`);
      out.push(lines[line - 1]!);
      used += lines[line - 1]!.length + 1;
    }
    if (line <= to) { next = line; break; }
    next = to + 1;
  }
  if (next <= lines.length) out.push(`[… lines ${next}–${lines.length} …]`);
  return out.join('\n');
}

/** The briefing as the agent reads it. Deterministic: an unchanged briefing renders the same text. */
export function briefingText(briefing: Briefing): string {
  // Stored documents drop empty lists, so a list may be absent.
  const list = (title: string, items: string[] | undefined) => items?.length ? [`${title}:`, ...items.map(item => `- ${item}`)] : [];
  return ['Notes from your harness companion, which watches your work in the background. Use what helps; ignore the rest.',
    `Focus: ${briefing.focus}`, ...list('Known', briefing.facts), ...list('Watch out', briefing.warnings),
    ...list('Consider', briefing.suggestions)].join('\n');
}

/** What the companion looked up while the agent's last reply streamed, as the agent reads it. */
export function researchText(notes: string[]): string {
  return ['Looked up while you were writing your last reply:', ...notes.map(note => `- ${note}`)].join('\n');
}

export type CompanionOptions = {
  /** The opened Harness, for starting companion tasks from a hook (hooks have no commit). */
  harness(): Harness;
  /** Called with each failure the companion keeps going after. */
  onReport?(error: unknown): void;
  /**
   * Pluggable hot path: how a long tool output is shortened. `crisp` (default): its head and tail; `nl`: `shape.nl`
   * chooses the lines and says what the rest holds; `shadow`: both, the natural-language shape shown and the two compared.
   */
  shaping?: PluggableSetting;
  /**
   * Pluggable hot path: what the agent's streamed reasoning and text name that the companion looks up before the turn
   * ends (stream.ts, plans/STREAMING.md §2). `crisp` (default): written paths and quoted names; `nl`: `hints.nl`
   * judges; `shadow`: both, the natural-language hints used and the two compared.
   */
  hints?: PluggableSetting;
  /** Called with each step of the companion's work on the live stream (stream.ts `SpeculationEvent`). */
  onSpeculation?(event: SpeculationEvent): void;
};

/** The companion extension; its functions run on `natlang`. */
export function companion(natlang: NatlangRuntime, options: CompanionOptions): Extension {
  const shaping = pluggableMode(options.shaping, 'crisp');
  const stream = new CompanionStream({ natlang, harness: options.harness, hints: options.hints,
    ...(options.onReport ? { onReport: options.onReport } : {}), ...(options.onSpeculation ? { onSpeculation: options.onSpeculation } : {}) });
  const task = defineTask<{ basis?: number }, { phase: 'observe' }, null, object>({
    name: TASK, version: 1, initial: () => ({ phase: 'observe' }),
    phases: {
      async observe(current, runtime, context) {
        const agent = await runtime.agent(context);
        const env = await runtime.env(context);
        const cwd = agent.cwd ?? env?.cwd ?? '/';
        const view = await runtime.context(runtime.conversationId, context);
        const messages = view.messages as Message[];
        const goal = [...messages].reverse().find(message => message.role === 'user');
        const recent = messages.slice(-RECENT_MESSAGES);
        const observation: Observation = JSON.parse(JSON.stringify({
          goal: goal ? renderMessage(goal).replace(/^USER: /, '') : '',
          recent: recent.map(renderMessage).filter(Boolean).join('\n'),
          touched: touched(recent, cwd),
          previous: (await runtime.snapshot(CompanionDoc, runtime.conversationId, context))?.briefing ?? null,
          known: Object.values((await runtime.snapshot(CompanionFiles, context))?.files ?? {}).slice(-40),
        }));
        let briefing: Briefing | undefined, failure: string | undefined;
        try {
          briefing = await natlang.run(() => observe(observation as never), {
            services: { companion: companionService(runtime, context, env, cwd, knowledge => runtime.commit(async tx => {
              (await tx.doc(CompanionFiles)).files[knowledge.path] = knowledge as never; return undefined; }, context)) },
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

  /**
   * The agent's view of a long output, by the shaping setting. The natural-language shape is memoized for the call, so
   * a rerun of the hook shows the same lines; when it fails, the crisp shape stands in and the failure is reported.
   */
  const shaped = async (call: ToolCall, text: string, api: HookApi, context: Context): Promise<string> => {
    const crisp = () => shapeOutput(text, call.id);
    const judged = async () => {
      const memo = `companion.shape.${call.id}`;
      const chosen = await api.memo<OutputShape>(memo, context) ?? await api.memo(memo, JSON.parse(JSON.stringify(
        await shape(`${call.name} ${JSON.stringify(call.arguments)}`, text, SHAPE_HEAD + SHAPE_TAIL))) as OutputShape, context);
      return renderShape(text, call.id, chosen);
    };
    const choose = pluggable({ crisp, nl: judged }, shaping, { name: 'companion.shape' });
    if (shaping === 'crisp') return choose();
    try {
      return await natlang.run(choose, { name: `companion.shape#${call.id}` });
    } catch (error) {
      options.onReport?.(error);
      return crisp();
    }
  };

  /**
   * Latency: the agent model's reader is declared at startup, so when it is Neuralese the view of a stored output is
   * written right away, in the background, for that reader only; the request that reads it joins or finds it
   * (host/views.ts forceView). A failure here is reported; the request forcing the call fails loudly if it persists.
   */
  const preforce = async (conversationId: ConversationId, call: string, models: Models) => {
    try {
      const context = BACKGROUND_CONTEXT;
      const harness = options.harness();
      const conversation = await harness.conversation(conversationId, context);
      const ref = conversation && (await conversation.agent(context)).model;
      const model = ref && models.getModel(ref.provider, ref.modelId);
      if (!conversation || !model || modelReader(model).kind !== 'neuralese') return;
      const owner = (await harness.snapshot(ProviderDoc, conversationId, context))?.sessionId;
      if (!owner) return;
      const docs: ViewDocs = { conversationId, read: harness, commit: (change, at) => conversation.commit(change, at) };
      await forceView(docs, call, { models, model, owner, store: natlang.options.neuralese?.store }, context);
    } catch (error) { options.onReport?.(error); }
  };

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
        // The raw output: recall forces nothing, whatever the reader.
        const handle = String((args as { handle?: unknown }).handle ?? '');
        const stored = await options.harness().snapshot(CompanionOutputs, api.conversationId, handle, context);
        if (!stored) return { content: [{ type: 'text', text: `No shortened output has the handle ${JSON.stringify(handle)}.` }], isError: true };
        return { content: [{ type: 'text', text: stored.text }] };
      },
    }],
    sections: [{
      key: 'companion',
      async render(input, context) {
        const doc = await input.read.snapshot(CompanionDoc, input.conversationId, context);
        const text = [doc?.briefing ? briefingText(doc.briefing) : '', doc?.research ? researchText(doc.research.notes) : ''].filter(Boolean);
        return text.length ? text.join('\n') : undefined;
      },
    }],
    hooks: [hook(ToolTask, {
      // A long output is kept whole in the companion's store and stored as a call of view (host/views.ts): the result
      // holds its text form, shaped when it is longer than SHAPE_LIMIT (COMPANION.md §1.3 and §6), with the reference.
      async afterTool(call, result, api, context) {
        const parts = result.content ?? [];
        const text = parts.map(part => part.type === 'text' ? part.text : '').join('');
        if (call.name === 'recall' || text.length <= VIEW_LIMIT) return undefined;
        try {
          const conversation = await options.harness().conversation(api.conversationId, context);
          await conversation?.commit(async tx => { await tx.doc(CompanionOutputs, api.conversationId, call.id, { tool: call.name, text }); }, context);
        } catch (error) { options.onReport?.(error); return undefined; }
        const shown = text.length > SHAPE_LIMIT ? await shaped(call, text, api, context) : text;
        void preforce(api.conversationId, call.id, api.models);
        return { ...result, content: [{ type: 'text', text: shown, stored: { function: 'view', call: call.id } } as never,
          ...parts.filter(part => part.type !== 'text')] };
      },
    }), hook(GenerationTask, {
      // The intent of each tool call, whatever the agent's model reads: a cheap text document, the instructions of the
      // view of the call's output (host/views.ts viewIntent, the harness bench's records.py intent()). Recorded for a
      // text reader too, so a conversation switched to a Neuralese reader can force the views of earlier calls;
      // forcing itself happens only for a Neuralese reader.
      // The intent of a call that completed while the reply streamed was captured then (stream.ts); the terminal message
      // confirms it, and settles the rest of the turn's speculative work.
      async afterResponse(message, api, context) {
        const captured = await stream.settle(message, api.conversationId, api.taskId, context);
        const calls = message.content.filter((part): part is ToolCall => part.type === 'toolCall');
        if (!calls.length) return;
        const conversation = await options.harness().conversation(api.conversationId, context);
        await conversation?.commit(async tx => {
          for (const call of calls) await tx.doc(ViewIntents, api.conversationId, call.id,
            { intent: captured.get(call.id) ?? viewIntent(message as AssistantMessage, call) });
        }, context);
      },
      // The agent's reply as it streams: hints for early research and tool-call preparation (stream.ts).
      onStream(event, attempt, api, context) { stream.observe(event, attempt, api, context); },
      // After every tool round: the agent has new information, and the companion catches up while it thinks.
      async afterTools(assistant, _results, api, context) { await start(api.conversationId, assistant as number, context); },
    })],
  };
}
