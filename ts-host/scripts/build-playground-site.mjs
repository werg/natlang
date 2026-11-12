#!/usr/bin/env node
/**
 * The public playground: a static copy of playground/ with the browser runtime, for hosting beside the website
 * (.github/workflows/pages.yml). It is the playground's public edition (`<html data-edition="public">`): no case or
 * training workbench and no local pipeline API. Natural-language functions run on the stock LFM3 1.7B Q4_K_M GGUF,
 * which the browser downloads from Hugging Face (it serves CORS and range requests; GitHub release assets do not).
 *
 *   node scripts/build-playground-site.mjs --out SITE    writes SITE/play/ (the page) and SITE/dist/browser/ (the runtime)
 *
 * Needs `npm run build` (the browser bundle and dist/model-catalog.js).
 */
import { cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { exitUnlessFreshBrowserBuild, STAMP } from './browser-build-freshness.mjs';

const root = resolve(import.meta.dirname, '..');
const outIndex = process.argv.indexOf('--out');
if (outIndex < 0 || !process.argv[outIndex + 1]) throw new Error('usage: build-playground-site.mjs --out SITE');
const site = resolve(process.argv[outIndex + 1]);
exitUnlessFreshBrowserBuild('scripts/build-playground-site.mjs');

// The runtime: exactly the files the browser build wrote or copied (its stamp), at the path app.mjs imports.
rmSync(resolve(site, 'dist/browser'), { recursive: true, force: true });
const stamp = JSON.parse(readFileSync(STAMP, 'utf8'));
for (const path of [...Object.keys(stamp.outputs), ...Object.keys(stamp.assets)]) {
  mkdirSync(dirname(resolve(site, path)), { recursive: true });
  cpSync(resolve(root, path), resolve(site, path));
}

const play = resolve(site, 'play');
rmSync(play, { recursive: true, force: true });
cpSync(resolve(root, 'playground'), play, { recursive: true, filter: path => !path.endsWith('README.md') });
const { STOCK_MODELS } = await import(pathToFileURL(resolve(root, 'dist/model-catalog.js')).href);
const model = STOCK_MODELS.find(item => item.quant === 'Q4_K_M');
mkdirSync(resolve(play, 'models/templates'), { recursive: true });
cpSync(resolve(root, '../models/templates', model.template), resolve(play, 'models/templates', model.template));
writeFileSync(resolve(play, 'models/browser-catalog.json'), `${JSON.stringify({
  schema: 'natlang.browser-model-catalog/1', defaultId: model.id, models: [{ id: model.id, label: model.label,
    trainingRun: `${model.source.repository}@${model.source.revision}`, file: model.file, quant: model.quant,
    bytes: model.bytes, sha256: model.sha256, contextTokens: model.contextTokens, url: model.downloadUrl,
    templateUrl: `./models/templates/${model.template}` }],
}, null, 2)}\n`);
const page = resolve(play, 'index.html');
writeFileSync(page, readFileSync(page, 'utf8').replace('<html lang="en">',
  '<html lang="en" data-edition="public" data-model-catalog="./models/browser-catalog.json">'));
process.stdout.write(`public playground: ${play}\n`);
