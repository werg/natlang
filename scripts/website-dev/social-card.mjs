// Renders website/og.png, the 1200x630 preview image that links to the site show, from the page's own headline
// and example. Run it after changing either: node scripts/website-dev/social-card.mjs [--out <file>]
// Options: --out <file>, --chromium <executable>. Uses ts-host's playwright-core and its Chromium
// (`npx playwright-core install chromium` in ts-host), or the executable given with --chromium.
import { createRequire } from 'node:module';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const { chromium } = createRequire(join(REPO, 'ts-host', 'package.json'))('playwright-core');

const option = name => { const i = process.argv.indexOf(name); return i > 0 ? process.argv[i + 1] : undefined; };
const out = resolve(option('--out') ?? join(REPO, 'website', 'og.png'));
const executablePath = option('--chromium');

// Keep the brand, the headline and the code plate; drop navigation, prose and captions.
const CARD_CSS = `
  header.bar .platedots, header.bar nav, header.bar button { display: none !important; }
  .htext .lead, .htext .actions, .htext .smallnote, .codeplate .plate-cap { display: none !important; }
  .home { padding-top: 84px !important; }
  .hero { align-items: start !important; margin-top: 0 !important; padding-top: 0 !important; }
  .codeplate pre.code { font-size: 11.5px !important; line-height: 1.5 !important; }
`;

const browser = await chromium.launch({ executablePath });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, colorScheme: 'light' });
  await page.goto(pathToFileURL(join(REPO, 'website', 'index.html')).href);
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: CARD_CSS });
  await page.screenshot({ path: out });
  console.log(`Wrote ${out}`);
} finally {
  await browser.close();
}
