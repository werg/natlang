/**
 * `natlang run examples/refunds [-- FILE.json]`: settle refund requests (a JSON file of them, or the bundled requests.json),
 * record them in the desk folder under the program's state directory, and print the outcomes and the ledger.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { openFolder, saveFolder, untrusted, type PluggableSetting, type TargetContext } from '@natlang/node';
import { settle } from './desk.js';
import { record } from './ledger.js';
import type { Request } from './types.js';

export async function main(context: TargetContext): Promise<number> {
  const root = context.package?.root ?? context.workspace;
  const read = (path: string) => readFileSync(join(root, path), 'utf8');
  const { file, eligibility } = context.arguments as { file?: string; eligibility?: PluggableSetting };
  // The customer's text enters here: mark it as outside data, labelled with where it came from.
  const requests = (JSON.parse(file ? readFileSync(file, 'utf8') : read('requests.json')) as Request[])
    .map(request => ({ ...request, text: untrusted(request.text, `customer request ${request.id}`) }));
  const policies = { outdoor: read('policies/outdoor.md'), books: read('policies/books.md') };
  const desk = join(context.stateDirectory, 'desk');
  mkdirSync(join(desk, 'cases'), { recursive: true });
  if (!existsSync(join(desk, 'ledger.json'))) writeFileSync(join(desk, 'ledger.json'), '{ "refund": 0, "partial": 0, "decline": 0 }\n');
  const folder = openFolder(desk, 'overlay');
  const report = await context.runtime.run(async () => {
    const outcomes = await settle(requests, { policies, ...(eligibility ? { eligibility } : {}) });
    return { outcomes, ledger: await record(folder, outcomes) ?? 'not recorded: the proposal changed other files' };
  });
  saveFolder(desk, folder);
  context.io.output.write(JSON.stringify(report, null, 2) + '\n');
  return 0;
}
