import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

const inventory = JSON.parse(readFileSync(process.argv[2], 'utf8'));
for (const [name, root] of Object.entries(inventory.roots)) {
  if (process.env[name] !== root.path) throw new Error(`${name} root differs`);
  const bytes = readFileSync(join(process.env[name], root.sample.path));
  if (createHash('sha256').update(bytes).digest('hex') !== root.sample.sha256) {
    throw new Error(`${name} case-referenced asset differs`);
  }
}
console.log(JSON.stringify({ rl_asset_node_reads: 'passed', roots: Object.keys(inventory.roots) }));
