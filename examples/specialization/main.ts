/**
 * A small program for watching trace-guided specialization work (plans/TRACE_SPECIALIZATION.md).
 *
 *   natlang run examples/specialization -- workload 40     run 40 generated requests through support.nl
 *   natlang run examples/specialization -- ask "refund 12" answer one request
 *
 * Every call is recorded in the machine's call store. `natlang specialize --definition support` then compiles the
 * recorded calls into cases; later calls whose inputs a case's guard admits are answered without the model.
 */
import { createNatlangRuntime, type TargetContext } from '@natlang/node';
import support from './support.nl';

/** An in-memory order service; its log shows which effects happened. */
export function orderService() {
  const log: string[] = [];
  return { log, orders: {
    async refund(number: string) { log.push(`refund ${number}`); return { ok: true, number }; },
    lookup(number: string) { log.push(`lookup ${number}`); return { number, status: Number(number) % 3 === 0 ? 'delivered' : Number(number) % 3 === 1 ? 'shipped' : 'packed' }; },
  } };
}

const TEMPLATES = [
  (n: number) => `refund ${n}`, (n: number) => `Please refund order ${n}`, (n: number) => `status ${n}`,
  (n: number) => `Where is order ${n}?`, (n: number) => `What is the status of order ${n}?`,
  () => 'Do you ship to Canada?', () => 'I was charged twice, can someone call me?', () => 'thanks!',
];

export async function main(context: TargetContext): Promise<number> {
  const [command = 'workload', value = '20'] = context.args;
  const service = orderService();
  const runtime = createNatlangRuntime({ model: context.model, services: { orders: service.orders },
    programRoot: context.package?.root ?? context.workspace });
  const requests = command === 'ask' ? [context.args.slice(1).join(' ')] :
    Array.from({ length: Number(value) }, (_, index) => TEMPLATES[index % TEMPLATES.length]!(100 + index));
  for (const request of requests) {
    const started = Date.now();
    const answer = await runtime.run(() => support(request)).catch(error => `error: ${error instanceof Error ? error.message : String(error)}`);
    context.io.output.write(`${request} -> ${answer} (${Date.now() - started} ms)\n`);
  }
  context.io.output.write(`effects: ${service.log.join(', ') || 'none'}\n`);
  return 0;
}
