import { execFileSync } from 'node:child_process';
import { mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

const tsHost = resolve(import.meta.dirname, '..');
const dist = resolve(tsHost, 'dist');
rmSync(dist, { recursive: true, force: true });

// Stamp the build with the commit it was built from ("dirty-<sha>" when ts-host has uncommitted changes); the
// data scripts (port_record_calls.sh, static_curriculum_campaign.py) record it. Later build steps keep dist/.
const git = (...args) => execFileSync('git', args, { cwd: tsHost, encoding: 'utf8' }).trim();
let stamp = 'unknown';
try {
  const sha = git('rev-parse', 'HEAD');
  stamp = git('status', '--porcelain', '--', '.') ? `dirty-${sha}` : sha;
} catch {}
mkdirSync(dist, { recursive: true });
writeFileSync(resolve(dist, '.build-commit'), `${stamp}\n`);
