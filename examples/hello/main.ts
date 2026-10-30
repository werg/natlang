import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { TargetContext } from '@natlang/node';
import answer from './answer.nl';

export async function main(context: TargetContext): Promise<number> {
  const sample = JSON.parse(readFileSync(join(context.package?.root ?? context.workspace, 'sample.json'), 'utf8'));
  const input = context.args[0] ?? sample.input;
  const result = await context.runtime.run(() => answer(input));
  context.io.output.write(JSON.stringify(result, null, 2) + '\n');
  return 0;
}
