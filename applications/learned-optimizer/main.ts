/**
 * `natlang run applications/learned-optimizer -- adapt PLAN.json OUT [--steps N] [--python PATH]`: one Natlang
 * adaptation episode over a prepared learned-optimizer tune plan (`natlang_neuralese.tune.learned.apply` plan); prints
 * the final state and the selected bundle's release fragment.
 * `natlang run applications/learned-optimizer -- train OPERATION.json [--cycles N] [--python PATH]`: scheduled
 * outer-training cycles over an operation (`TrainingServices` fields).
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { TargetContext } from '@natlang/node';
import { AdaptationServices, TrainingServices, adapt, train, type PythonEntry } from './index.js';

export async function main(context: TargetContext): Promise<number> {
  const args = context.args;
  const option = (name: string) => { const index = args.indexOf(name); return index >= 0 ? args[index + 1] : undefined; };
  const positional = args.filter((arg, index) => !arg.startsWith('--') && !args[index - 1]?.startsWith('--'));
  const python: PythonEntry = { argv: [option('--python') ?? 'python3', '-m'], cwd: context.workspace,
    env: { PYTHONPATH: [resolve(context.workspace, 'training/neuralese'), context.workspace, process.env.PYTHONPATH].filter(Boolean).join(':') } };
  const [command, ...rest] = positional;
  if (command === 'adapt' && rest.length === 2) {
    const services = new AdaptationServices(python, resolve(context.workspace, rest[0]!), resolve(context.workspace, rest[1]!));
    const state = await adapt(context.runtime, services, Number(option('--steps') ?? 4));
    context.io.output.write(`${JSON.stringify(state, null, 2)}\n`);
    context.io.output.write(`${JSON.stringify(await services.release(state.current?.id ?? null), null, 2)}\n`);
    return state.current ? 0 : 1;
  }
  if (command === 'train' && rest.length === 1) {
    const operation = JSON.parse(readFileSync(resolve(context.workspace, rest[0]!), 'utf8'));
    const state = await train(context.runtime, new TrainingServices(python, operation), Number(option('--cycles') ?? 1));
    context.io.output.write(`${JSON.stringify(state, null, 2)}\n`);
    return 0;
  }
  context.io.error.write('usage: adapt PLAN.json OUT [--steps N] | train OPERATION.json [--cycles N]  [--python PATH]\n');
  return 2;
}
