/**
 * The Chromium the browser smokes launch: NATLANG_CHROMIUM, else the build pinned by playwright-core, else the newest
 * Chromium Playwright has installed (a newer build than the pinned one, or an arm64 build under chrome-linux/).
 */
import { existsSync, readdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';

export function chromiumPath() {
  if (process.env.NATLANG_CHROMIUM) return process.env.NATLANG_CHROMIUM;
  if (existsSync(chromium.executablePath())) return chromium.executablePath();
  const cache = process.env.PLAYWRIGHT_BROWSERS_PATH ?? join(homedir(), '.cache/ms-playwright');
  const versions = existsSync(cache) ? readdirSync(cache).filter(name => /^chromium-\d+$/.test(name))
    .sort((a, b) => Number(b.slice(9)) - Number(a.slice(9))) : [];
  for (const version of versions) for (const dir of ['chrome-linux64', 'chrome-linux']) {
    const candidate = join(cache, version, dir, 'chrome');
    if (existsSync(candidate)) return candidate;
  }
  throw new Error('Chromium is missing; install it with npx playwright install chromium, or set NATLANG_CHROMIUM');
}
