/** The shared ask pipeline for the shell command and the command-line entry point. */
import { Folder } from './scoped-fs.js';

export type AskOptions = { lines?: boolean; jsonl?: boolean; files?: string; filter?: boolean;
  jobs?: number; returns?: string };
export type AskHost = (instruction: string, returns: string, input: unknown) => Promise<unknown>;
export type CallMode = 'whole' | 'line' | 'jsonl';
export type CallHost = (name: string, input: unknown, mode: CallMode) => Promise<unknown>;

export function parseAskArgs(args: string[]): { instruction: string; options: AskOptions } {
  const options: AskOptions = {}, words: string[] = [];
  for (let index = 0; index < args.length; index++) {
    const arg = args[index]!;
    if (arg === '--lines') options.lines = true;
    else if (arg === '--jsonl') options.jsonl = true;
    else if (arg === '--filter') options.filter = true;
    else if (arg === '--returns' || arg === '--files' || arg === '--jobs') {
      const value = args[++index];
      if (value === undefined) throw new Error(`${arg} needs a value`);
      if (arg === '--returns') options.returns = value;
      else if (arg === '--files') options.files = value;
      else options.jobs = Number(value);
    } else if (arg.startsWith('--')) throw new Error(`unknown natlang ask option: ${arg}`);
    else words.push(arg);
  }
  if (!words.length) throw new Error('natlang ask needs an instruction');
  return { instruction: words.join(' '), options };
}

function format(value: unknown): string {
  return value === null || typeof value === 'object' ? JSON.stringify(value) : String(value);
}

/** The same JSON/line input contract for the CLI and the folder shell. */
export async function executeNatlangCall(name: string, stdin: string, mode: CallMode, call: CallHost): Promise<string> {
  const inputs = mode === 'whole' ? [stdin.trim() ? JSON.parse(stdin) : {}] :
    stdin.split(/\r?\n/).filter(line => line.length).map(line => mode === 'jsonl' ? JSON.parse(line) : line);
  const values: string[] = [];
  for (const input of inputs) values.push(format(await call(name, input, mode)));
  return values.length ? `${values.join('\n')}\n` : '';
}

/** Apply one instruction to stdin, each line/record, or each matched file, preserving input order. */
export async function executeNatlangAsk(folder: Folder, instruction: string, stdin: string,
  options: AskOptions, ask: AskHost): Promise<string> {
  if (options.lines && options.jsonl) throw new Error('--lines and --jsonl are mutually exclusive');
  if (options.files && (options.lines || options.jsonl)) throw new Error('--files cannot be combined with --lines or --jsonl');
  const jobs = options.jobs ?? 1;
  if (!Number.isSafeInteger(jobs) || jobs < 1 || jobs > 64) throw new RangeError('--jobs must be between 1 and 64');
  const returns = options.returns ?? (options.filter ? 'boolean' : 'string');
  type Item = { input: unknown; display: string; path?: string };
  let items: Item[];
  if (options.files) items = (await folder.root().files(options.files)).map(file =>
    ({ input: file, display: file.path, path: file.path }));
  else if (options.lines || options.jsonl) items = stdin.split(/\r?\n/).filter(line => line.length).map(line =>
    ({ input: options.jsonl ? JSON.parse(line) : line, display: line }));
  else items = [{ input: stdin ? stdin : folder.root(), display: stdin }];
  const results = new Array<unknown>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(jobs, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await ask(instruction, returns, items[index]!.input);
    }
  }));
  const lines = items.flatMap((item, index) => {
    const result = results[index];
    if (options.filter) return result === true ? [item.display] : [];
    return [item.path ? `${item.path}\t${format(result)}` : format(result)];
  });
  return lines.length ? `${lines.join('\n')}\n` : '';
}
