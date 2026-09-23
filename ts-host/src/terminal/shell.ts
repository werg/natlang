import { createInterface } from 'node:readline/promises';
import type { Readable, Writable } from 'node:stream';
import { EventQueue, type AppEvent, type EventLoop } from '../app/event-loop.js';
import { renderTerminalView, type TerminalView } from './view.js';

export type TerminalShellOptions<E extends AppEvent> = {
  input?: Readable; output?: Writable; color?: boolean; width?: number;
  prompt?: string; event: (line: string, id: string) => E;
  events?: AsyncIterable<E>;
  cancelEvent?: (id: string) => E;
  commands?: Record<string, { description: string; run(args: string): Promise<string | E | E[] | void> | string | E | E[] | void }>;
};

/** Interactive line transport. Slash commands control presentation/lifecycle only. */
export async function runTerminalShell<S, E extends AppEvent>(
  app: EventLoop<S, TerminalView, E>, options: TerminalShellOptions<E>): Promise<void> {
  const input = options.input ?? process.stdin, output = options.output ?? process.stdout;
  const show = (view: TerminalView): void => {
    output.write(renderTerminalView(view, {
      color: options.color ?? Boolean((output as NodeJS.WriteStream).isTTY),
      width: options.width ?? (output as NodeJS.WriteStream).columns ?? 80,
    }));
  };
  show((await app.start()).view);
  const lines = createInterface({ input, output, terminal: Boolean((output as NodeJS.WriteStream).isTTY) });
  const queue = new EventQueue<E>();
  let sequence = 0;
  const writeNote = (text: string): void => { output.write(text.trimEnd() + '\n\n'); };
  const help = (): void => {
    const rows = ['/help show commands', '/refresh redraw', '/interrupt abort the current step',
      '/cancel request application cancellation', '/quit exit',
      ...Object.entries(options.commands ?? {}).map(([name, command]) => `/${name} ${command.description}`),
      ...(app.view?.help ?? [])];
    writeNote([...new Set(rows)].join('\n'));
  };
  const consume = app.consume(queue, transition => show(transition.view), error =>
    writeNote(`Request failed: ${error instanceof Error ? error.message : String(error)}`));
  const external = options.events ? (async () => {
    try { for await (const event of options.events!) queue.push(event); }
    catch (error) { queue.fail(error); }
  })() : Promise.resolve();
  // The iterator buffers lines that arrive during a step (piped input) and ends at EOF.
  const received = lines[Symbol.asyncIterator]();
  try {
    while (true) {
      lines.setPrompt(app.view?.prompt ?? options.prompt ?? '> ');
      lines.prompt();
      const next = await received.next();
      if (next.done) break;
      const value = next.value.trim();
      if (!value) continue;
      if (value === '/quit' || value === '/exit') break;
      if (value === '/help') { help(); continue; }
      if (value === '/refresh') { show((await app.refresh()).view); continue; }
      if (value === '/interrupt') { app.cancel(); continue; }
      if (value === '/cancel') {
        if (options.cancelEvent) queue.push(options.cancelEvent(
          `terminal-${Date.now().toString(36)}-${++sequence}`));
        else app.cancel();
        continue;
      }
      if (value.startsWith('/')) {
        const space = value.indexOf(' '), name = value.slice(1, space < 0 ? undefined : space);
        const command = options.commands?.[name];
        if (!command) { writeNote(`Unknown command /${name}. Use /help.`); continue; }
        try {
          const result = await command.run(space < 0 ? '' : value.slice(space + 1).trim());
          if (typeof result === 'string') writeNote(result);
          else if (result) for (const item of Array.isArray(result) ? result : [result]) queue.push(item);
        } catch (error) { writeNote(`/${name}: ${error instanceof Error ? error.message : String(error)}`); }
        continue;
      }
      queue.push(options.event(value, `terminal-${Date.now().toString(36)}-${++sequence}`));
    }
  } finally {
    lines.close(); queue.close(); await consume; await app.close();
    void external;
  }
}
