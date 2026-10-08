/**
 * pi's coding tools as one extension, "coding-tools": read, write, edit and bash, in that order (the order is
 * model-visible). The declarations are pi's, verbatim, because requests compare them at every turn (§7.4). Each tool's
 * procedure is a natural-language function in this folder, run in its own natlang task with the call's `env` service;
 * a function that fails becomes a thrown error, which the tool task turns into pi's tool_error result.
 */
import type { Context } from '@earendil-works/chord';
import type { NatlangRuntime } from '@natlang/node';
import { Type } from 'typebox';
import type { Extension, ToolExecutionApi, ToolExecutionResult, ToolRegistration } from '../../vendor/durable/src/harness/types.ts';
import { ENV_DECLARATION, envService } from '../../host/env.ts';
import read from './read.nl';
import write from './write.nl';
import edit from './edit.nl';
import bash from './bash.nl';
import prepareEditArguments from './edit/prepare.ts';

const DEFAULT_MAX_LINES = 2000;
const DEFAULT_MAX_BYTES = 50 * 1024;

const readSchema = Type.Object({
  path: Type.String({ description: 'Path to the file to read (relative or absolute)' }),
  offset: Type.Optional(Type.Number({ description: 'Line number to start reading from (1-indexed)' })),
  limit: Type.Optional(Type.Number({ description: 'Maximum number of lines to read' })),
});

const writeSchema = Type.Object({
  path: Type.String({ description: 'Path to the file to write (relative or absolute)' }),
  content: Type.String({ description: 'Content to write to the file' }),
});

const replaceEditSchema = Type.Object({
  oldText: Type.String({ description: 'Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.' }),
  newText: Type.String({ description: 'Replacement text for this targeted edit.' }),
});

const editSchema = Type.Object({
  path: Type.String({ description: 'Path to the file to edit (relative or absolute)' }),
  edits: Type.Array(replaceEditSchema, { description: 'One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.' }),
});

const bashSchema = Type.Object({
  command: Type.String({ description: 'Bash command to execute' }),
  timeout: Type.Optional(Type.Number({ description: 'Timeout in seconds (optional, no default timeout)' })),
});

/**
 * The reason of a natural-language call that finished `failed` (NatlangCallError detail "error: <reason>") or
 * `blocked` ("blocked: <reason>"), as the tool's error; anything else (an abort, an infrastructure failure) unchanged.
 */
export function toolError(error: unknown, signal: AbortSignal | undefined): unknown {
  if (signal?.aborted) return error;
  const record = error as { name?: unknown; detail?: unknown };
  if (record?.name !== 'NatlangCallError' || typeof record.detail !== 'string') return error;
  const reason = /^(?:error|blocked): ([\s\S]*)$/.exec(record.detail);
  return reason ? new Error(reason[1]) : error;
}

/** Run a tool's natural-language function with the call's env service; locks and readers it left are released. */
export async function runToolFunction(natlang: NatlangRuntime, name: string, fn: (args: never) => Promise<unknown>, args: unknown,
  api: ToolExecutionApi, context: Context): Promise<ToolExecutionResult> {
  const env = envService(api, context);
  const signal = context.abortSignal;
  try {
    return await natlang.run(() => fn(args as never), { services: { env }, serviceDeclarations: { env: ENV_DECLARATION },
      ...(signal ? { signal } : {}), name: `tool:${name}#${api.callId}` }) as ToolExecutionResult;
  } catch (error) { throw toolError(error, signal); }
  finally { await env.release(); }
}

/** The coding tools, each run by `natlang`. */
export function codingTools(natlang: NatlangRuntime): Extension {
  const tools: ToolRegistration[] = [
    {
      name: 'read',
      description: `Read the contents of a text file. Output is truncated to ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.`,
      parameters: readSchema,
      execute: (args, api, context) => runToolFunction(natlang, 'read', read as never, args, api, context),
    },
    {
      name: 'write',
      description: "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.",
      parameters: writeSchema,
      execute: (args, api, context) => runToolFunction(natlang, 'write', write as never, args, api, context),
    },
    {
      name: 'edit',
      description: 'Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.',
      parameters: editSchema,
      prepareArguments: args => prepareEditArguments(args) as never,
      execute: (args, api, context) => runToolFunction(natlang, 'edit', edit as never, args, api, context),
    },
    {
      name: 'bash',
      description: `Execute a bash command in the current working directory. Returns combined stdout and stderr. Output is truncated to last ${DEFAULT_MAX_LINES} lines or ${DEFAULT_MAX_BYTES / 1024}KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.`,
      parameters: bashSchema,
      outputLimits: { retain: 'tail' },
      execute: (args, api, context) => runToolFunction(natlang, 'bash', bash as never, args, api, context),
    },
  ] as ToolRegistration[];
  return { name: 'coding-tools', tools };
}
