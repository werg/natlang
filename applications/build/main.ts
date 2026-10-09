/**
 * `natlang run applications/build -- GRAPH.json [--goal ID] [--root DIR] [--ready P] [--choose P] [--validity P]`:
 * build the goal of a task graph. GRAPH.json is `{ "goal": "id", "tasks": [{ id, needs, description, argv, inputs,
 * outputs }] }`; --goal overrides its goal and --root (default: the workspace) is where tasks run. Each policy point
 * (ready, choose, validity) runs its `crisp` or `natural-language` implementation. Exit status 0 when the goal is done.
 */
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openFolder, type TargetContext } from '@natlang/node';
import { BuildWorkspace, buildGoal, type Implementation, type PolicyPoint, type Task } from './index.js';

const POINTS: PolicyPoint[] = ['ready', 'choose', 'validity'];

export async function main(context: TargetContext): Promise<number> {
  const args = context.args;
  const option = (name: string) => args.find((_, i) => args[i - 1] === name);
  const path = args.find((arg, i) => !arg.startsWith('--') && !args[i - 1]?.startsWith('--'));
  if (!path) {
    context.io.error.write('usage: GRAPH.json [--goal ID] [--root DIR] [--ready|--choose|--validity crisp|natural-language]\n');
    return 2;
  }
  const spec = JSON.parse(readFileSync(resolve(context.workspace, path), 'utf8')) as { goal?: string, tasks: Task[] };
  const goal = option('--goal') ?? spec.goal;
  if (!goal || !Array.isArray(spec.tasks)) { context.io.error.write('the graph needs a goal and a list of tasks\n'); return 2; }
  const policy: Partial<Record<PolicyPoint, Implementation>> = {};
  for (const point of POINTS) {
    const choice = option(`--${point}`);
    if (choice === undefined) continue;
    if (choice !== 'crisp' && choice !== 'natural-language') { context.io.error.write(`--${point} takes crisp or natural-language\n`); return 2; }
    policy[point] = choice;
  }
  const root = resolve(context.workspace, option('--root') ?? '.');
  const workspace = await new BuildWorkspace(root, { policy }).open();
  const report = await buildGoal(context.runtime, workspace, goal, spec.tasks, openFolder(root).root());
  for (const event of workspace.drainEvents()) context.io.error.write(`${JSON.stringify(event)}\n`);
  context.io.output.write(`${report.status}: ${report.summary}\n`);
  for (const line of report.next) context.io.output.write(`  - ${line}\n`);
  context.io.output.write(`${JSON.stringify(report, null, 2)}\n`);
  return report.status === 'done' ? 0 : 1;
}
